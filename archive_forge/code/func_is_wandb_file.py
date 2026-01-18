import os
from typing import Callable, Generator, Union
def is_wandb_file(name: str) -> bool:
    return name.startswith('wandb') or name == METADATA_FNAME or name == CONFIG_FNAME or (name == REQUIREMENTS_FNAME) or (name == OUTPUT_FNAME) or (name == DIFF_FNAME) or (name == CONDA_ENVIRONMENTS_FNAME)