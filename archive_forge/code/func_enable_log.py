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
def enable_log(self, log_qnames, log_level):
    if isinstance(log_qnames, str):
        log_qnames = [log_qnames]
    for log_qname in log_qnames:
        self.log_qname_to_level[log_qname] = log_level