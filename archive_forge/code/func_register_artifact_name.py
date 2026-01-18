import functools
import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import __import__
from typing import Dict, List, Optional, Set, Union
from weakref import WeakSet
import torch._guards
import torch.distributed as dist
def register_artifact_name(self, name, description, visible, off_by_default, log_format):
    self.artifact_names.add(name)
    if visible:
        self.visible_artifacts.add(name)
    self.artifact_descriptions[name] = description
    if off_by_default:
        self.off_by_default_artifact_names.add(name)
    if log_format is not None:
        self.artifact_log_formatters[name] = logging.Formatter(log_format)