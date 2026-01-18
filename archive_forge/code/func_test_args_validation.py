import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def test_args_validation(self):
    if self.march() == 'unknown':
        return
    for baseline, dispatch in (('unkown_feature - max +min', 'unknown max min'), ('#avx2', '$vsx')):
        try:
            self.nopt(cpu_baseline=baseline, cpu_dispatch=dispatch)
            raise AssertionError('excepted an exception for invalid arguments')
        except DistutilsError:
            pass