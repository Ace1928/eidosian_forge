import os
from typing import Callable, Generator, Union
def exclude_wandb_fn(path: str, root: str) -> bool:
    return any((os.path.relpath(path, root).startswith(wandb_dir + os.sep) for wandb_dir in WANDB_DIRS))