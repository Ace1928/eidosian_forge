import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
def atomic_save(state_dict: Any, path: str) -> None:
    """
    Like torch.save, but atomic.

    Useful for preventing trouble coming from being pre-empted or killed while writing
    to disk. Works by writing to a temporary file, and then renaming the file to the
    final name.
    """
    torch.save(state_dict, path + '.tmp')
    os.rename(path + '.tmp', path)