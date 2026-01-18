import argparse
import functools
import itertools
import marshal
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List
def indent_msg(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args[0].indent += 1
        ret = fn(*args, **kwargs)
        args[0].indent -= 1
        return ret
    return wrapper