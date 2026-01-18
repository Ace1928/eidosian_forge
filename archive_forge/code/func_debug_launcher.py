import os
import sys
import tempfile
import torch
from .state import AcceleratorState, PartialState
from .utils import (
def debug_launcher(function, args=(), num_processes=2):
    """
    Launches a training function using several processes on CPU for debugging purposes.

    <Tip warning={true}>

    This function is provided for internal testing and debugging, but it's not intended for real trainings. It will
    only use the CPU.

    </Tip>

    Args:
        function (`Callable`):
            The training function to execute.
        args (`Tuple`):
            Tuple of arguments to pass to the function (it will receive `*args`).
        num_processes (`int`, *optional*, defaults to 2):
            The number of processes to use for training.
    """
    from torch.multiprocessing import start_processes
    with tempfile.NamedTemporaryFile() as tmp_file:
        with patch_environment(world_size=num_processes, master_addr='127.0.0.1', master_port='29500', accelerate_mixed_precision='no', accelerate_debug_rdv_file=tmp_file.name, accelerate_use_cpu='yes'):
            launcher = PrepareForLaunch(function, debug=True)
            start_processes(launcher, args=args, nprocs=num_processes, start_method='fork')