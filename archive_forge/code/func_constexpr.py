import binascii
import hashlib
import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List
import triton
from triton.compiler.code_generator import kernel_suffix
from triton.compiler.make_launcher import ty_to_cpp
def constexpr(s):
    try:
        ret = int(s)
        return ret
    except ValueError:
        pass
    try:
        ret = float(s)
        return ret
    except ValueError:
        pass
    return None