import sys
import tempfile
import torch
def init_torch_distributed_local():
    if torch.distributed.is_initialized():
        return
    init_url = 'file://' + tempfile.mkstemp()[1]
    backend = torch.distributed.Backend.NCCL if torch.cuda.is_available() and (not is_windows) else torch.distributed.Backend.GLOO
    torch.distributed.init_process_group(backend=backend, rank=0, world_size=1, init_method=init_url)