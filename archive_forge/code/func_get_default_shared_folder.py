import argparse
import os
from pathlib import Path
from xformers.benchmarks.LRA.run_tasks import Task
from xformers.components.attention import ATTENTION_REGISTRY
def get_default_shared_folder() -> str:
    checkpoint_paths = ['/checkpoint', '/checkpoints']
    for checkpoint_path in checkpoint_paths:
        if Path(checkpoint_path).is_dir():
            return checkpoint_path
    return '.'