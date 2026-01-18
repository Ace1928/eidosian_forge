import argparse
import os
import uuid
from pathlib import Path
import submitit
from xformers.benchmarks.LRA.run_tasks import benchmark, get_arg_parser
def get_shared_folder() -> Path:
    user = os.getenv('USER')
    checkpoint_paths = ['/checkpoint', '/checkpoints']
    for checkpoint_path in checkpoint_paths:
        if Path(checkpoint_path).is_dir():
            p = Path(f'{checkpoint_path}/{user}/xformers/submitit')
            p.mkdir(exist_ok=True, parents=True)
            return p
    raise RuntimeError(f'No shared folder available - considering {checkpoint_paths}')