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
def block_stmt(stmt: str, indent: int=0) -> str:
    """Partially unroll benchmark loop.

            The naive template looks something like:
                "for _ in range({number}): {stmt}"

            However a loop in Python is surprisingly expensive, and significantly
            increases the number of background Python instructions. So instead we
            partially unroll the loops, with a block size of 100 chosen to keep
            the instruction overhead from `range` low while also not ballooning
            the size of the generated file.
            """
    block_size = 100
    loop_count = number // block_size
    if loop_count == 1:
        loop_count = 0
    remainder = number - block_size * loop_count
    blocked_stmt = ''
    if loop_count:
        unrolled_stmts = textwrap.indent('\n'.join([stmt] * block_size), ' ' * 4)
        blocked_stmt += f'for _ in range({loop_count}):\n{unrolled_stmts}\n'
    if remainder:
        blocked_stmt += '\n'.join([stmt] * remainder)
    return textwrap.indent(blocked_stmt, ' ' * indent)