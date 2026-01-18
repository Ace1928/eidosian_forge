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
def grab_method_and_template(in_kernel):
    pos = {'kernel_launch': {'start': in_kernel['start'], 'end': in_kernel['end']}, 'kernel_name': {'start': -1, 'end': -1}, 'template': {'start': -1, 'end': -1}}
    count = {'<>': 0}
    START = 0
    AT_TEMPLATE = 1
    AFTER_TEMPLATE = 2
    AT_KERNEL_NAME = 3
    status = START
    for i in range(pos['kernel_launch']['start'] - 1, -1, -1):
        char = string[i]
        if status in (START, AT_TEMPLATE):
            if char == '>':
                if status == START:
                    status = AT_TEMPLATE
                    pos['template']['end'] = i
                count['<>'] += 1
            if char == '<':
                count['<>'] -= 1
                if count['<>'] == 0 and status == AT_TEMPLATE:
                    pos['template']['start'] = i
                    status = AFTER_TEMPLATE
        if status != AT_TEMPLATE:
            if string[i].isalnum() or string[i] in {'(', ')', '_', ':', '#'}:
                if status != AT_KERNEL_NAME:
                    status = AT_KERNEL_NAME
                    pos['kernel_name']['end'] = i
                if i == 0:
                    pos['kernel_name']['start'] = 0
                    return [pos['kernel_name'], pos['template'], pos['kernel_launch']]
            elif status == AT_KERNEL_NAME:
                pos['kernel_name']['start'] = i
                return [pos['kernel_name'], pos['template'], pos['kernel_launch']]