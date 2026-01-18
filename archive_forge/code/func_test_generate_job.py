import os
import numpy as np
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm.base as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.interfaces.base import traits
def test_generate_job(create_files_in_directory):

    class TestClass(spm.SPMCommand):
        input_spec = spm.SPMCommandInputSpec
    dc = TestClass()
    out = dc._generate_job()
    assert out == ''
    contents = {'contents': [1, 2, 3, 4]}
    out = dc._generate_job(contents=contents)
    assert out == '.contents(1) = 1;\n.contents(2) = 2;\n.contents(3) = 3;\n.contents(4) = 4;\n'
    filelist, outdir = create_files_in_directory
    names = spm.scans_for_fnames(filelist, keep4d=True)
    contents = {'files': names}
    out = dc._generate_job(prefix='test', contents=contents)
    assert out == "test.files = {...\n'a.nii';...\n'b.nii';...\n};\n"
    contents = 'foo'
    out = dc._generate_job(prefix='test', contents=contents)
    assert out == "test = 'foo';\n"
    contents = {'onsets': np.array((1,), dtype=object)}
    contents['onsets'][0] = [1, 2, 3, 4]
    out = dc._generate_job(prefix='test', contents=contents)
    assert out == 'test.onsets = {...\n[1, 2, 3, 4];...\n};\n'