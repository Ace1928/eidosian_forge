import collections
import enum
import dataclasses
import itertools as it
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
from typing import (
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import CallgrindModuleType
def denoise(self) -> 'FunctionCounts':
    """Remove known noisy instructions.

        Several instructions in the CPython interpreter are rather noisy. These
        instructions involve unicode to dictionary lookups which Python uses to
        map variable names. FunctionCounts is generally a content agnostic
        container, however this is sufficiently important for obtaining
        reliable results to warrant an exception."""
    return self.filter(lambda fn: 'dictobject.c:lookdict_unicode' not in fn)