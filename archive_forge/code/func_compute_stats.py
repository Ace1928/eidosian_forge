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
def compute_stats(stats):
    unsupported_calls = {cuda_call for cuda_call, _filepath in stats['unsupported_calls']}
    print(f'Total number of unsupported CUDA function calls: {len(unsupported_calls):d}')
    print(', '.join(unsupported_calls))
    print(f'\nTotal number of replaced kernel launches: {len(stats['kernel_launches']):d}')