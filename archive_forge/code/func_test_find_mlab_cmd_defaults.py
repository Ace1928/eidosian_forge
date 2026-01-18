import os
import numpy as np
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm.base as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.interfaces.base import traits
def test_find_mlab_cmd_defaults():
    saved_env = dict(os.environ)

    class TestClass(spm.SPMCommand):
        pass
    for varname in ['FORCE_SPMMCR', 'SPMMCRCMD']:
        try:
            del os.environ[varname]
        except KeyError:
            pass
    dc = TestClass()
    assert dc._use_mcr is None
    assert dc._matlab_cmd is None
    os.environ['FORCE_SPMMCR'] = '1'
    dc = TestClass()
    assert dc._use_mcr
    assert dc._matlab_cmd is None
    os.environ['SPMMCRCMD'] = 'spmcmd'
    dc = TestClass()
    assert dc._use_mcr
    assert dc._matlab_cmd == 'spmcmd'
    os.environ.clear()
    os.environ.update(saved_env)