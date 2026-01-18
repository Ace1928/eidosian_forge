import logging
import datetime
import time
import ray
import cupy
from ray.util.collective.const import ENV
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
from ray.util.collective.collective_group.cuda_stream import get_stream_pool
def collective_fn(input_tensor, output_tensor, comm, stream):
    comm.reduceScatter(nccl_util.get_tensor_ptr(input_tensor), nccl_util.get_tensor_ptr(output_tensor), nccl_util.get_tensor_n_elements(output_tensor), nccl_util.get_nccl_tensor_dtype(output_tensor), nccl_util.get_nccl_reduce_op(reducescatter_options.reduceOp), stream.ptr)