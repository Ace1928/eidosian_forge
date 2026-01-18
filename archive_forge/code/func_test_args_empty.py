import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def test_args_empty(self):
    for baseline, dispatch in (('', 'none'), (None, ''), ('none +none', 'none - none'), ('none -max', 'min - max'), ('+vsx2 -VSX2', 'vsx avx2 avx512f -max'), ('max -vsx - avx + avx512f neon -MAX ', 'min -min + max -max -vsx + avx2 -avx2 +NONE')):
        opt = self.nopt(cpu_baseline=baseline, cpu_dispatch=dispatch)
        assert len(opt.cpu_baseline_names()) == 0
        assert len(opt.cpu_dispatch_names()) == 0