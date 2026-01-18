import sys
from io import BytesIO
from timeit import timeit
import numpy as np
from ..fileslice import fileslice
from ..openers import ImageOpener
from ..optpkg import optional_package
from ..rstutils import rst_table
from ..tmpdirs import InTemporaryDirectory
def my_table(title, times, base):
    print()
    print(rst_table(times, ROW_NAMES, COL_NAMES, title, val_fmt='{0[0]:3.2f} ({0[1]:3.2f})'))
    print(f'Base time: {base:3.2f}')