from typing import List, Optional, Tuple, Union
import pyarrow as pa
import pyhdk
from packaging import version
from pyhdk.hdk import HDK, ExecutionResult, QueryNode, RelAlgExecutor
from modin.config import CpuCount, HdkFragmentSize, HdkLaunchParameters
from modin.utils import _inherit_docstrings
from .base_worker import BaseDbWorker, DbTable
@classmethod
def executeRA(cls, query: str, exec_calcite=False, **exec_args):
    hdk = cls._hdk()
    if exec_calcite or query.startswith('execute calcite'):
        ra = hdk._calcite.process(query, db_name='hdk', legacy_syntax=True)
    else:
        ra = query
    ra_executor = RelAlgExecutor(hdk._executor, hdk._schema_mgr, hdk._data_mgr, ra)
    table = ra_executor.execute(device_type=cls._preferred_device, **exec_args)
    return HdkTable(table)