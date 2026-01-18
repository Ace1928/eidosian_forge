import asyncio
import time
from typing import List
import numpy
import ray
import ray.experimental.internal_kv as internal_kv
from ray._raylet import GcsClient
from ray.util.collective.types import ReduceOp, torch_available
from ray.util.queue import _QueueActor
def get_gloo_reduce_op(reduce_op):
    """Map the reduce op to GLOO reduce op type.

    Args:
        reduce_op: ReduceOp Enum (SUM/PRODUCT/MIN/MAX).

    Returns:
        (pygloo.ReduceOp): the mapped GLOO reduce op.
    """
    if reduce_op not in GLOO_REDUCE_OP_MAP:
        raise RuntimeError("Gloo does not support reduce op: '{}'.".format(reduce_op))
    return GLOO_REDUCE_OP_MAP[reduce_op]