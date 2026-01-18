import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def hw_prob_pop_func(self):
    return _hw_func(self.stream, False, True)