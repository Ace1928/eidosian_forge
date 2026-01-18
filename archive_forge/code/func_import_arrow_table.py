from typing import List, Optional, Tuple, Union
import pyarrow as pa
import pyhdk
from packaging import version
from pyhdk.hdk import HDK, ExecutionResult, QueryNode, RelAlgExecutor
from modin.config import CpuCount, HdkFragmentSize, HdkLaunchParameters
from modin.utils import _inherit_docstrings
from .base_worker import BaseDbWorker, DbTable
@classmethod
def import_arrow_table(cls, table: pa.Table, name: Optional[str]=None):
    name = cls._genName(name)
    table = cls.cast_to_compatible_types(table, _CAST_DICT)
    hdk = cls._hdk()
    fragment_size = cls.compute_fragment_size(table)
    return HdkTable(hdk.import_arrow(table, name, fragment_size))