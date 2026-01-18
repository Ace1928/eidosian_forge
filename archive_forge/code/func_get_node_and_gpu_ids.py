import pickle
from typing import Optional, List, Tuple, TYPE_CHECKING
from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip, set_cuda_visible_devices, get_ip
def get_node_and_gpu_ids(self) -> Tuple[str, List[int]]:
    node_id = ray.get_runtime_context().get_node_id()
    gpu_ids = ray.get_gpu_ids()
    return (node_id, gpu_ids)