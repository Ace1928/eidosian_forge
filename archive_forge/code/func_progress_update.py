import itertools
import platform
import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import click
import wandb
from . import ipython, sparkline
def progress_update(self, text: str, percent_done: float) -> None:
    if self._progress:
        self._progress.update(percent_done, text)