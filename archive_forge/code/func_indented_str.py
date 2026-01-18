import contextlib
import dataclasses
import math
import textwrap
from typing import Any, Dict, Optional
import torch
from torch import inf
def indented_str(s, indent):
    return '\n'.join((f'  {line}' for line in s.split('\n')))