from __future__ import absolute_import
import sys
import os
def exec_file(program_name, args=()):
    runcmd([os.path.abspath(program_name)] + list(args), shell=False)