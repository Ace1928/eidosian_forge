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
def register_artifact_log(self, artifact_log_qname):
    self.artifact_log_qnames.add(artifact_log_qname)