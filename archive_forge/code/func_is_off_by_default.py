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
def is_off_by_default(self, artifact_qname):
    return artifact_qname in self.off_by_default_artifact_names