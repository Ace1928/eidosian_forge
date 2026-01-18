import os
import sys
from setuptools import setup
def process_template_file(contents):
    ret = []
    template_lines = []
    append_to = ret
    for line in contents.splitlines(keepends=False):
        if line.strip() == '### TEMPLATE_START':
            append_to = template_lines
        elif line.strip() == '### TEMPLATE_END':
            append_to = ret
            for line in process_template_lines(template_lines):
                ret.append(line)
        else:
            append_to.append(line)
    return '\n'.join(ret)