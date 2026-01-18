import math
from enum import IntEnum
from typing import TYPE_CHECKING
import torch
from . import ir
from .utils import get_dtype_size, sympy_product
from .virtualized import V
def get_collective_type(snode: 'BaseSchedulerNode') -> NCCL_COLL:
    if isinstance(snode.node, (ir.AllReduce, ir.AllReduceCoalesced)):
        return NCCL_COLL.ALL_REDUCE
    elif isinstance(snode.node, (ir.AllGatherIntoTensor, ir.AllGatherIntoTensorCoalesced)):
        return NCCL_COLL.ALL_GATHER
    elif isinstance(snode.node, (ir.ReduceScatterTensor, ir.ReduceScatterTensorCoalesced)):
        return NCCL_COLL.REDUCE_SCATTER
    else:
        raise Exception(f'Unsupported collective type: {snode.node}')