import re
import warnings
from numba.core import typing, sigutils
from numba.pycc.compiler import ExportEntry
def process_input_files(inputs):
    """
    Read input source files for execution of legacy @export / @exportmany
    decorators.
    """
    for ifile in inputs:
        with open(ifile) as fin:
            exec(compile(fin.read(), ifile, 'exec'))