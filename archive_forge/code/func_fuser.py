import contextlib
from typing import List, Tuple
import torch
@contextlib.contextmanager
def fuser(name):
    """Context manager that facilitates switching between backend fusers.

    Valid names:
    * ``fuser0`` - enables only legacy fuser
    * ``fuser1`` - enables only NNC
    * ``fuser2`` - enables only nvFuser
    * ``fuser3`` - enables oneDNN Graph
    """
    old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
    old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
    old_texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
    old_nvfuser_state = torch._C._jit_nvfuser_enabled()
    old_llga_state = torch._C._jit_llga_enabled()
    if name == 'fuser0':
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    elif name == 'fuser1':
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        old_profiling_mode = torch._C._get_graph_executor_optimize(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    elif name == 'fuser2':
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._jit_set_llga_enabled(False)
    elif name == 'fuser3':
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        old_profiling_mode = torch._C._get_graph_executor_optimize(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(True)
    elif name == 'none':
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    else:
        raise Exception(f'unrecognized fuser option (name: {name})')
    try:
        yield
    finally:
        if name in ['fuser1', 'fuser3']:
            torch._C._jit_set_profiling_executor(old_profiling_executor)
            torch._C._get_graph_executor_optimize(old_profiling_mode)
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(old_gpu_fuse)
        torch._C._jit_set_texpr_fuser_enabled(old_texpr_fuser_state)
        torch._C._jit_set_nvfuser_enabled(old_nvfuser_state)
        torch._C._jit_set_llga_enabled(old_llga_state)