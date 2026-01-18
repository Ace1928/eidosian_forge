import pickle
from typing import Optional, List, Tuple, TYPE_CHECKING
from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip, set_cuda_visible_devices, get_ip
def get_node_ip(self) -> str:
    return get_ip()