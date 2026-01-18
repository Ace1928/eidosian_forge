import itertools
import platform
import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import click
import wandb
from . import ipython, sparkline
def progress_close(self, _: Optional[str]=None) -> None:
    if self._progress:
        self._progress.close()