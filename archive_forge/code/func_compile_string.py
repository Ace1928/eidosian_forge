import argparse
import functools
import itertools
import marshal
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List
def compile_string(self, file_content: str) -> types.CodeType:
    path_marker = PATH_MARKER
    return compile(file_content, path_marker, 'exec')