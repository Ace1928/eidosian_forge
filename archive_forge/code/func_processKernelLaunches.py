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
def processKernelLaunches(string, stats):
    """ Replace the CUDA style Kernel launches with the HIP style kernel launches."""
    string = RE_KERNEL_LAUNCH.sub(lambda inp: f'{inp.group(1)}{inp.group(2)}::', string)

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

    def find_kernel_bounds(string):
        """Finds the starting and ending points for all kernel launches in the string."""
        kernel_end = 0
        kernel_positions = []
        while string.find('<<<', kernel_end) != -1:
            kernel_start = string.find('<<<', kernel_end)
            kernel_end = string.find('>>>', kernel_start) + 3
            if kernel_end <= 0:
                raise InputError('no kernel end found')
            kernel_positions.append({'start': kernel_start, 'end': kernel_end, 'group': string[kernel_start:kernel_end]})
        return kernel_positions

    def mask_comments(string):
        in_comment = ''
        prev_c = ''
        new_string = ''
        for c in string:
            if in_comment == '':
                if c == '/' and prev_c == '/':
                    in_comment = '//'
                elif c == '*' and prev_c == '/':
                    in_comment = '/*'
                elif c == '"' and prev_c != '\\' and (prev_c != "'"):
                    in_comment = '"'
            elif in_comment == '//':
                if c == '\r' or c == '\n':
                    in_comment = ''
            elif in_comment == '/*':
                if c == '/' and prev_c == '*':
                    in_comment = ''
            elif in_comment == '"':
                if c == '"' and prev_c != '\\':
                    in_comment = ''
            prev_c = c
            if in_comment == '':
                new_string += c
            else:
                new_string += 'x'
        return new_string
    get_kernel_positions = list(find_kernel_bounds(mask_comments(string)))
    output_string = string
    for kernel in get_kernel_positions:
        params = grab_method_and_template(kernel)
        parenthesis = string.find('(', kernel['end'])
        cuda_kernel = string[params[0]['start']:parenthesis + 1]
        kernel_string = string[kernel['start']:kernel['end']]
        end_param_index = 0 if params[1]['end'] == -1 else 1
        kernel_name_with_template = string[params[0]['start']:params[end_param_index]['end'] + 1]
        cuda_kernel_dim3 = add_dim3(kernel_string, cuda_kernel)
        num_klp = len(extract_arguments(0, kernel['group'].replace('<<<', '(').replace('>>>', ')')))
        hip_kernel = 'hipLaunchKernelGGL(' + cuda_kernel_dim3[0:-1].replace('>>>', ', 0' * (4 - num_klp) + '>>>').replace('<<<', ', ').replace('>>>', ', ').replace(kernel_name_with_template, '(' + kernel_name_with_template + ')')
        output_string = output_string.replace(cuda_kernel, hip_kernel)
        stats['kernel_launches'].append(hip_kernel)
    return output_string