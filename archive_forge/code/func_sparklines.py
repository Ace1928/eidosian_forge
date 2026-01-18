import itertools
import platform
import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import click
import wandb
from . import ipython, sparkline
def sparklines(self, series: List[Union[int, float]]) -> Optional[str]:
    if wandb.util.is_unicode_safe(sys.stdout):
        return sparkline.sparkify(series)
    return None