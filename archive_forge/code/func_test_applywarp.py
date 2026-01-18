import os
from copy import deepcopy
import pytest
import pdb
from nipype.utils.filemanip import split_filename, ensure_list
from .. import preprocess as fsl
from nipype.interfaces.fsl import Info
from nipype.interfaces.base import File, TraitError, Undefined, isdefined
from nipype.interfaces.fsl import no_fsl
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_applywarp(setup_flirt):
    tmpdir, infile, reffile = setup_flirt
    opt_map = {'out_file': ('--out=bar.nii', 'bar.nii'), 'premat': ('--premat=%s' % reffile, reffile), 'postmat': ('--postmat=%s' % reffile, reffile)}
    for name, settings in list(opt_map.items()):
        awarp = fsl.ApplyWarp(in_file=infile, ref_file=reffile, field_file=reffile, **{name: settings[1]})
        if name == 'out_file':
            realcmd = 'applywarp --in=%s --ref=%s --out=%s --warp=%s' % (infile, reffile, settings[1], reffile)
        else:
            outfile = awarp._gen_fname(infile, suffix='_warp')
            realcmd = 'applywarp --in=%s --ref=%s --out=%s --warp=%s %s' % (infile, reffile, outfile, reffile, settings[0])
        assert awarp.cmdline == realcmd