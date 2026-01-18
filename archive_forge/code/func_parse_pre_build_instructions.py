from __future__ import (absolute_import, division, print_function)
import resource
import base64
import contextlib
import errno
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
def parse_pre_build_instructions(requirements):
    """Parse the given pip requirements and return a list of extracted pre-build instructions."""
    pre_build_prefix = '# pre-build '
    pre_build_requirement_prefix = pre_build_prefix + 'requirement: '
    pre_build_constraint_prefix = pre_build_prefix + 'constraint: '
    lines = requirements.splitlines()
    pre_build_lines = [line for line in lines if line.startswith(pre_build_prefix)]
    instructions = []
    for line in pre_build_lines:
        if line.startswith(pre_build_requirement_prefix):
            instructions.append(PreBuild(line[len(pre_build_requirement_prefix):]))
        elif line.startswith(pre_build_constraint_prefix):
            instructions[-1].constraints.append(line[len(pre_build_constraint_prefix):])
        else:
            raise RuntimeError('Unsupported pre-build comment: ' + line)
    return instructions