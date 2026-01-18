import argparse
import logging
import os
import sys
import pythran
import pythran.types.tog
from distutils.errors import CompileError
def convert_arg_line_to_args(arg_line):
    """Read argument from file in a prettier way."""
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg