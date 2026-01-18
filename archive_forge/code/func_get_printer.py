import itertools
import platform
import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import click
import wandb
from . import ipython, sparkline
def get_printer(_jupyter: bool) -> Printer:
    if _jupyter:
        return PrinterJupyter()
    return PrinterTerm()