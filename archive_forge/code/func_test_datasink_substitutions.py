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
def test_datasink_substitutions(tmpdir):
    indir = tmpdir.mkdir('-Tmp-nipype_ds_subs_in')
    outdir = tmpdir.mkdir('-Tmp-nipype_ds_subs_out')
    files = []
    for n in ['ababab.n', 'xabababyz.n']:
        f = str(indir.join(n))
        files.append(f)
        open(f, 'w')
    ds = nio.DataSink(parameterization=False, base_directory=str(outdir), substitutions=[('ababab', 'ABABAB')], regexp_substitutions=[('xABABAB(\\w*)\\.n$', 'a-\\1-b.n'), ('(.*%s)[-a]([^%s]*)$' % ((os.path.sep,) * 2), '\\1!\\2')])
    setattr(ds.inputs, '@outdir', files)
    ds.run()
    assert sorted([os.path.basename(x) for x in glob.glob(os.path.join(str(outdir), '*'))]) == ['!-yz-b.n', 'ABABAB.n']