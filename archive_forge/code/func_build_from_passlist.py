from functools import wraps
from inspect import unwrap
from typing import Callable, List, Optional
import logging
@classmethod
def build_from_passlist(cls, passes):
    pm = PassManager(passes)
    return pm