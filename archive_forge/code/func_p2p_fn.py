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
def p2p_fn(tensor, comm, stream, peer):
    comm.recv(nccl_util.get_tensor_ptr(tensor), recv_options.n_elements if recv_options.n_elements > 0 else nccl_util.get_tensor_n_elements(tensor), nccl_util.get_nccl_tensor_dtype(tensor), peer, stream.ptr)