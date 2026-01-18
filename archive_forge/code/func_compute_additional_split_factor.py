import math
from typing import Optional, Tuple, Union
from ray import available_resources as ray_available_resources
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.util import _autodetect_parallelism
from ray.data.context import WARN_PREFIX, DataContext
from ray.data.datasource.datasource import Datasource, Reader
def compute_additional_split_factor(datasource_or_legacy_reader: Union[Datasource, Reader], parallelism: int, mem_size: int, target_max_block_size: int, cur_additional_split_factor: Optional[int]=None) -> Tuple[int, str, int, Optional[int]]:
    ctx = DataContext.get_current()
    parallelism, reason, _, _ = _autodetect_parallelism(parallelism, target_max_block_size, ctx, datasource_or_legacy_reader, mem_size)
    num_read_tasks = len(datasource_or_legacy_reader.get_read_tasks(parallelism))
    expected_block_size = None
    if mem_size:
        expected_block_size = mem_size / num_read_tasks
        logger.get_logger().debug(f'Expected in-memory size {mem_size}, block size {expected_block_size}')
        size_based_splits = round(max(1, expected_block_size / target_max_block_size))
    else:
        size_based_splits = 1
    if cur_additional_split_factor:
        size_based_splits *= cur_additional_split_factor
    logger.get_logger().debug(f'Size based split factor {size_based_splits}')
    estimated_num_blocks = num_read_tasks * size_based_splits
    logger.get_logger().debug(f'Blocks after size splits {estimated_num_blocks}')
    available_cpu_slots = ray_available_resources().get('CPU', 1)
    if parallelism and num_read_tasks >= available_cpu_slots * 4 and (num_read_tasks >= 5000):
        logger.get_logger().warn(f'{WARN_PREFIX} The requested parallelism of {parallelism} is more than 4x the number of available CPU slots in the cluster of {available_cpu_slots}. This can lead to slowdowns during the data reading phase due to excessive task creation. Reduce the parallelism to match with the available CPU slots in the cluster, or set parallelism to -1 for Ray Data to automatically determine the parallelism. You can ignore this message if the cluster is expected to autoscale.')
    if estimated_num_blocks < parallelism and estimated_num_blocks > 0:
        k = math.ceil(parallelism / estimated_num_blocks)
        estimated_num_blocks = estimated_num_blocks * k
        return (parallelism, reason, estimated_num_blocks, k)
    return (parallelism, reason, estimated_num_blocks, None)