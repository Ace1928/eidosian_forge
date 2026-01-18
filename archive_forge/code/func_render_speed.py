import io
import sys
import typing
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from io import RawIOBase, UnsupportedOperation
from math import ceil
from mmap import mmap
from operator import length_hint
from os import PathLike, stat
from threading import Event, RLock, Thread
from types import TracebackType
from typing import (
from . import filesize, get_console
from .console import Console, Group, JustifyMethod, RenderableType
from .highlighter import Highlighter
from .jupyter import JupyterMixin
from .live import Live
from .progress_bar import ProgressBar
from .spinner import Spinner
from .style import StyleType
from .table import Column, Table
from .text import Text, TextType
@classmethod
def render_speed(cls, speed: Optional[float]) -> Text:
    """Render the speed in iterations per second.

        Args:
            task (Task): A Task object.

        Returns:
            Text: Text object containing the task speed.
        """
    if speed is None:
        return Text('', style='progress.percentage')
    unit, suffix = filesize.pick_unit_and_suffix(int(speed), ['', '×10³', '×10⁶', '×10⁹', '×10¹²'], 1000)
    data_speed = speed / unit
    return Text(f'{data_speed:.1f}{suffix} it/s', style='progress.percentage')