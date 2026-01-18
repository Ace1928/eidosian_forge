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
def get_log_qnames(self) -> Set[str]:
    return {qname for qnames in self.log_alias_to_log_qnames.values() for qname in qnames}