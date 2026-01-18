from abc import ABC, abstractmethod
from ctypes import ArgumentError
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from transformers.utils import is_tf_available
from ..base import ExportConfig
@classmethod
def get_mandatory_axes_for_task(cls, task: str) -> Tuple[str]:
    axes = []
    for axis in cls.MANDATORY_AXES:
        if isinstance(axis, tuple):
            tasks, name = axis
            if not isinstance(tasks, tuple):
                tasks = (tasks,)
            if task not in tasks:
                continue
        else:
            name = axis
        axes.append(name)
    return tuple(axes)