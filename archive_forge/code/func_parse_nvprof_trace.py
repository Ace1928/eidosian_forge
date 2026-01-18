from collections import defaultdict
from typing import Any, Dict, List, Optional
from warnings import warn
import torch
import torch.cuda
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import _ExperimentalConfig
from torch.autograd import (
from torch.autograd.profiler_util import (
from torch.futures import Future
def parse_nvprof_trace(path):
    import sqlite3
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    strings = {}
    for r in conn.execute('SELECT _id_ as id, value FROM StringTable'):
        strings[r['id']] = torch._C._demangle(r['value'])
    marker_query = '\n    SELECT\n        start.id AS marker_id, start.name, start.timestamp AS start_time, end.timestamp AS end_time\n    FROM\n        CUPTI_ACTIVITY_KIND_MARKER AS start INNER JOIN CUPTI_ACTIVITY_KIND_MARKER AS end\n        ON start.id = end.id\n    WHERE\n        start.name != 0 AND end.name = 0\n    '
    functions = []
    functions_map = {}
    unique = EnforceUnique()
    for row in conn.execute(marker_query):
        unique.see(row['marker_id'])
        evt = FunctionEvent(id=row['marker_id'], node_id=0, name=strings[row['name']], start_us=row['start_time'], end_us=row['end_time'], thread=0)
        functions.append(evt)
        functions_map[evt.id] = evt
    kernel_query = '\n    SELECT\n        start.id AS marker_id, start.name, start.timestamp, end.timestamp,\n        runtime._id_ AS runtime_id, runtime.cbid, runtime.start AS runtime_start, runtime.end AS runtime_end,\n        kernel.start AS kernel_start, kernel.end AS kernel_end, kernel.name AS kernel_name\n    FROM\n        CUPTI_ACTIVITY_KIND_MARKER AS start\n        INNER JOIN CUPTI_ACTIVITY_KIND_MARKER AS end\n            ON start.id = end.id\n        INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME as runtime\n            ON (start.timestamp < runtime.start AND runtime.end < end.timestamp)\n        INNER JOIN CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL AS kernel\n            ON kernel.correlationId = runtime.correlationId\n    '
    unique = EnforceUnique()
    for row in conn.execute(kernel_query):
        unique.see(row['marker_id'], row['runtime_id'])
        assert row['cbid'] == 211
        evt = functions_map[row['marker_id']]
        evt.append_kernel(row['kernel_name'], 0, row['kernel_end'] - row['kernel_start'])
    functions.sort(key=lambda evt: evt.time_range.start)
    return functions