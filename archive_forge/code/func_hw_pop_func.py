import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def hw_pop_func(self):
    return _read_table(self.stream, [str, _gp_float, _gp_float, _gp_float])