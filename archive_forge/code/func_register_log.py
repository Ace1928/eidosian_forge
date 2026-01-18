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
def register_log(self, alias, log_qnames: Union[str, List[str]]):
    if isinstance(log_qnames, str):
        log_qnames = [log_qnames]
    self.log_alias_to_log_qnames[alias] = log_qnames