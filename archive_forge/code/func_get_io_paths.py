import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def get_io_paths(fname_inp, mname='untitled'):
    """Takes in a temporary file for testing and returns the expected output and input paths

    Here expected output is essentially one of any of the possible generated
    files.

    ..note::

         Since this does not actually run f2py, none of these are guaranteed to
         exist, and module names are typically incorrect

    Parameters
    ----------
    fname_inp : str
                The input filename
    mname : str, optional
                The name of the module, untitled by default

    Returns
    -------
    genp : NamedTuple PPaths
            The possible paths which are generated, not all of which exist
    """
    bpath = Path(fname_inp)
    return PPaths(finp=bpath.with_suffix('.f'), f90inp=bpath.with_suffix('.f90'), pyf=bpath.with_suffix('.pyf'), wrap77=bpath.with_name(f'{mname}-f2pywrappers.f'), wrap90=bpath.with_name(f'{mname}-f2pywrappers2.f90'), cmodf=bpath.with_name(f'{mname}module.c'))