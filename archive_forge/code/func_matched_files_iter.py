import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from typing import Dict, List, Iterator, Optional
from collections.abc import Mapping, Iterable
from enum import Enum
def matched_files_iter(root_path: str, includes: Iterable=(), ignores: Iterable=(), extensions: Iterable=(), out_of_place_only: bool=False, is_pytorch_extension: bool=False) -> Iterator[str]:
    exact_matches = set(includes)
    for abs_dirpath, dirs, filenames in os.walk(root_path, topdown=True):
        rel_dirpath = os.path.relpath(abs_dirpath, root_path)
        if rel_dirpath == '.':
            if '.git' in dirs:
                dirs.remove('.git')
            if 'build' in dirs:
                dirs.remove('build')
            if 'third_party' in dirs:
                dirs.remove('third_party')
                dirs.append('third_party/nvfuser')
        for filename in filenames:
            filepath = os.path.join(abs_dirpath, filename)
            rel_filepath = os.path.join(rel_dirpath, filename)
            if _fnmatch(filepath, includes) and (not _fnmatch(filepath, ignores)) and (match_extensions(filepath, extensions) or filepath in exact_matches):
                if not is_pytorch_extension:
                    if not is_pytorch_file(rel_filepath) and (not is_caffe2_gpu_file(rel_filepath)):
                        continue
                    if out_of_place_only and (not is_out_of_place(rel_filepath)):
                        continue
                yield filepath