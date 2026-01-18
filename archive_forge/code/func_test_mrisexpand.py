import os
import os.path as op
import pytest
from nipype.testing.fixtures import (
from nipype.pipeline import engine as pe
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.base import TraitError
from nipype.interfaces.io import FreeSurferSource
@pytest.mark.skipif(fs.no_freesurfer(), reason='freesurfer is not installed')
def test_mrisexpand(tmpdir):
    fssrc = FreeSurferSource(subjects_dir=fs.Info.subjectsdir(), subject_id='fsaverage', hemi='lh')
    fsavginfo = fssrc.run().outputs.get()
    expand_if = fs.MRIsExpand(in_file=fsavginfo['smoothwm'], out_name='expandtmp', distance=1, dt=60)
    expand_nd = pe.Node(fs.MRIsExpand(in_file=fsavginfo['smoothwm'], out_name='expandtmp', distance=1, dt=60), name='expand_node')
    orig_cmdline = 'mris_expand -T 60 {} 1 expandtmp'.format(fsavginfo['smoothwm'])
    assert expand_if.cmdline == orig_cmdline
    assert expand_nd.interface.cmdline == orig_cmdline
    nd_res = expand_nd.run()
    node_cmdline = 'mris_expand -T 60 -pial {cwd}/lh.pial {cwd}/lh.smoothwm 1 expandtmp'.format(cwd=nd_res.runtime.cwd)
    assert nd_res.runtime.cmdline == node_cmdline
    if_out_file = expand_if._list_outputs()['out_file']
    nd_out_file = nd_res.outputs.get()['out_file']
    assert op.basename(if_out_file) == op.basename(nd_out_file)
    assert op.dirname(if_out_file) == op.dirname(fsavginfo['smoothwm'])
    assert op.dirname(nd_out_file) == nd_res.runtime.cwd