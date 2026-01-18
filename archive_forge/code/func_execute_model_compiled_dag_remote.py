import pickle
from typing import Optional, List, Tuple, TYPE_CHECKING
from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip, set_cuda_visible_devices, get_ip
def execute_model_compiled_dag_remote(self, ignored):
    """Used only when compiled DAG is enabled."""
    import torch
    if not self.compiled_dag_cuda_device_set:
        torch.cuda.set_device(self.worker.device)
        self.compiled_dag_cuda_device_set = True
    output = self.worker.execute_model()
    output = pickle.dumps(output)
    return output