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
def test_fast(setup_infile):
    tmp_infile, tp_dir = setup_infile
    faster = fsl.FAST()
    faster.inputs.verbose = True
    fasted = fsl.FAST(in_files=tmp_infile, verbose=True)
    fasted2 = fsl.FAST(in_files=[tmp_infile, tmp_infile], verbose=True)
    assert faster.cmd == 'fast'
    assert faster.inputs.verbose
    assert faster.inputs.manual_seg == Undefined
    assert faster.inputs != fasted.inputs
    assert fasted.cmdline == 'fast -v -S 1 %s' % tmp_infile
    assert fasted2.cmdline == 'fast -v -S 2 %s %s' % (tmp_infile, tmp_infile)
    faster = fsl.FAST()
    faster.inputs.in_files = tmp_infile
    assert faster.cmdline == 'fast -S 1 %s' % tmp_infile
    faster.inputs.in_files = [tmp_infile, tmp_infile]
    assert faster.cmdline == 'fast -S 2 %s %s' % (tmp_infile, tmp_infile)
    opt_map = {'number_classes': ('-n 4', 4), 'bias_iters': ('-I 5', 5), 'bias_lowpass': ('-l 15', 15), 'img_type': ('-t 2', 2), 'init_seg_smooth': ('-f 0.035', 0.035), 'segments': ('-g', True), 'init_transform': ('-a %s' % tmp_infile, '%s' % tmp_infile), 'other_priors': ('-A %s %s %s' % (tmp_infile, tmp_infile, tmp_infile), ['%s' % tmp_infile, '%s' % tmp_infile, '%s' % tmp_infile]), 'no_pve': ('--nopve', True), 'output_biasfield': ('-b', True), 'output_biascorrected': ('-B', True), 'no_bias': ('-N', True), 'out_basename': ('-o fasted', 'fasted'), 'use_priors': ('-P', True), 'segment_iters': ('-W 14', 14), 'mixel_smooth': ('-R 0.25', 0.25), 'iters_afterbias': ('-O 3', 3), 'hyper': ('-H 0.15', 0.15), 'verbose': ('-v', True), 'manual_seg': ('-s %s' % tmp_infile, '%s' % tmp_infile), 'probability_maps': ('-p', True)}
    for name, settings in list(opt_map.items()):
        faster = fsl.FAST(in_files=tmp_infile, **{name: settings[1]})
        assert faster.cmdline == ' '.join([faster.cmd, settings[0], '-S 1 %s' % tmp_infile])