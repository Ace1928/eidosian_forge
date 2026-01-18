import os
import copy
import simplejson
import glob
import os.path as op
from subprocess import Popen
import hashlib
from collections import namedtuple
import pytest
import nipype
import nipype.interfaces.io as nio
from nipype.interfaces.base.traits_extension import isdefined
from nipype.interfaces.base import Undefined, TraitError
from nipype.utils.filemanip import dist_is_editable
from subprocess import check_call, CalledProcessError
def test_datafinder_depth(tmpdir):
    outdir = tmpdir.strpath
    os.makedirs(os.path.join(outdir, '0', '1', '2', '3'))
    df = nio.DataFinder()
    df.inputs.root_paths = os.path.join(outdir, '0')
    for min_depth in range(4):
        for max_depth in range(min_depth, 4):
            df.inputs.min_depth = min_depth
            df.inputs.max_depth = max_depth
            result = df.run()
            expected = ['{}'.format(x) for x in range(min_depth, max_depth + 1)]
            for path, exp_fname in zip(result.outputs.out_paths, expected):
                _, fname = os.path.split(path)
                assert fname == exp_fname