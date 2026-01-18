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
def hash_signature(signature: List[str]):
    m = hashlib.sha256()
    m.update(' '.join(signature).encode())
    return m.hexdigest()[:8]