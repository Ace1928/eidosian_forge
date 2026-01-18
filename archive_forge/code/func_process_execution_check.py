import contextlib
import io
import math
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.data_loader import SeedableRandomSampler, prepare_data_loader
from accelerate.state import AcceleratorState
from accelerate.test_utils import RegressionDataset, are_the_same_tensors
from accelerate.utils import (
def process_execution_check():
    accelerator = Accelerator()
    num_processes = accelerator.num_processes
    path = Path('check_main_process_first.txt')
    with accelerator.main_process_first():
        if accelerator.is_main_process:
            time.sleep(0.1)
            with open(path, 'a+') as f:
                f.write('Currently in the main process\n')
        else:
            with open(path, 'a+') as f:
                f.write('Now on another process\n')
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(path) as f:
            text = ''.join(f.readlines())
        try:
            assert text.startswith('Currently in the main process\n'), 'Main process was not first'
            if num_processes > 1:
                assert text.endswith('Now on another process\n'), 'Main process was not first'
            assert text.count('Now on another process\n') == accelerator.num_processes - 1, f'Only wrote to file {text.count('Now on another process') + 1} times, not {accelerator.num_processes}'
        except AssertionError:
            path.unlink()
            raise
    if accelerator.is_main_process and path.exists():
        path.unlink()
    accelerator.wait_for_everyone()
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        accelerator.on_main_process(print_main)(accelerator.state)
    result = f.getvalue().rstrip()
    if accelerator.is_main_process:
        assert result == 'Printing from the main process 0', f'{result} != Printing from the main process 0'
    else:
        assert f.getvalue().rstrip() == '', f'{result} != ""'
    f.truncate(0)
    f.seek(0)
    with contextlib.redirect_stdout(f):
        accelerator.on_local_main_process(print_local_main)(accelerator.state)
    if accelerator.is_local_main_process:
        assert f.getvalue().rstrip() == 'Printing from the local main process 0'
    else:
        assert f.getvalue().rstrip() == ''
    f.truncate(0)
    f.seek(0)
    with contextlib.redirect_stdout(f):
        accelerator.on_last_process(print_last)(accelerator.state)
    if accelerator.is_last_process:
        assert f.getvalue().rstrip() == f'Printing from the last process {accelerator.state.num_processes - 1}'
    else:
        assert f.getvalue().rstrip() == ''
    f.truncate(0)
    f.seek(0)
    for process_idx in range(num_processes):
        with contextlib.redirect_stdout(f):
            accelerator.on_process(print_on, process_index=process_idx)(accelerator.state, process_idx)
        if accelerator.process_index == process_idx:
            assert f.getvalue().rstrip() == f'Printing from process {process_idx}: {accelerator.process_index}'
        else:
            assert f.getvalue().rstrip() == ''
        f.truncate(0)
        f.seek(0)