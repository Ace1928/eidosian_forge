import os
from glob import glob
import shutil
from distutils.command.build_clib import build_clib as old_build_clib
from distutils.errors import DistutilsSetupError, DistutilsError, \
from numpy.distutils import log
from distutils.dep_util import newer_group
from numpy.distutils.misc_util import (
from numpy.distutils.ccompiler_opt import new_ccompiler_opt
def assemble_flags(self, in_flags):
    """ Assemble flags from flag list

        Parameters
        ----------
        in_flags : None or sequence
            None corresponds to empty list.  Sequence elements can be strings
            or callables that return lists of strings. Callable takes `self` as
            single parameter.

        Returns
        -------
        out_flags : list
        """
    if in_flags is None:
        return []
    out_flags = []
    for in_flag in in_flags:
        if callable(in_flag):
            out_flags += in_flag(self)
        else:
            out_flags.append(in_flag)
    return out_flags