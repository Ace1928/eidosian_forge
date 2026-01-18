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
def postprocess_fn(stream):
    for i, tensor_list in enumerate(tensor_lists):
        for j, tensor in enumerate(tensor_list):
            nccl_util.copy_tensor(tensor, output_flattened[i][j])