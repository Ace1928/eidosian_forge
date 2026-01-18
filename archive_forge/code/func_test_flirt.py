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
def test_flirt(setup_flirt):
    tmpdir, infile, reffile = setup_flirt
    flirter = fsl.FLIRT()
    assert flirter.cmd == 'flirt'
    flirter.inputs.bins = 256
    flirter.inputs.cost = 'mutualinfo'
    flirted = fsl.FLIRT(in_file=infile, reference=reffile, out_file='outfile', out_matrix_file='outmat.mat', bins=256, cost='mutualinfo')
    flirt_est = fsl.FLIRT(in_file=infile, reference=reffile, out_matrix_file='outmat.mat', bins=256, cost='mutualinfo')
    assert flirter.inputs != flirted.inputs
    assert flirted.inputs != flirt_est.inputs
    assert flirter.inputs.bins == flirted.inputs.bins
    assert flirter.inputs.cost == flirt_est.inputs.cost
    realcmd = 'flirt -in %s -ref %s -out outfile -omat outmat.mat -bins 256 -cost mutualinfo' % (infile, reffile)
    assert flirted.cmdline == realcmd
    flirter = fsl.FLIRT()
    with pytest.raises(ValueError):
        flirter.cmdline
    flirter.inputs.in_file = infile
    with pytest.raises(ValueError):
        flirter.cmdline
    flirter.inputs.reference = reffile
    pth, fname, ext = split_filename(infile)
    outfile = fsl_name(flirter, '%s_flirt' % fname)
    outmat = '%s_flirt.mat' % fname
    realcmd = 'flirt -in %s -ref %s -out %s -omat %s' % (infile, reffile, outfile, outmat)
    assert flirter.cmdline == realcmd
    axfm = deepcopy(flirter)
    axfm.inputs.apply_xfm = True
    with pytest.raises(RuntimeError):
        axfm.cmdline
    axfm2 = deepcopy(axfm)
    axfm.inputs.uses_qform = True
    assert axfm.cmdline == realcmd + ' -applyxfm -usesqform'
    axfm2.inputs.in_matrix_file = reffile
    assert axfm2.cmdline == realcmd + ' -applyxfm -init %s' % reffile
    tmpfile = tmpdir.join('file4test.nii')
    tmpfile.open('w')
    for key, trait_spec in sorted(fsl.FLIRT.input_spec().traits().items()):
        if key in ('trait_added', 'trait_modified', 'in_file', 'reference', 'environ', 'output_type', 'out_file', 'out_matrix_file', 'in_matrix_file', 'apply_xfm', 'resource_monitor', 'out_log', 'save_log'):
            continue
        param = None
        value = None
        if key == 'args':
            param = '-v'
            value = '-v'
        elif isinstance(trait_spec.trait_type, File):
            value = tmpfile.strpath
            param = trait_spec.argstr % value
        elif trait_spec.default is False:
            param = trait_spec.argstr
            value = True
        elif key in ('searchr_x', 'searchr_y', 'searchr_z'):
            value = [-45, 45]
            param = trait_spec.argstr % ' '.join((str(elt) for elt in value))
        else:
            value = trait_spec.default
            param = trait_spec.argstr % value
        cmdline = 'flirt -in %s -ref %s' % (infile, reffile)
        pth, fname, ext = split_filename(infile)
        outfile = fsl_name(fsl.FLIRT(), '%s_flirt' % fname)
        outfile = ' '.join(['-out', outfile])
        outmatrix = '%s_flirt.mat' % fname
        outmatrix = ' '.join(['-omat', outmatrix])
        cmdline = ' '.join([cmdline, outfile, outmatrix, param])
        flirter = fsl.FLIRT(in_file=infile, reference=reffile)
        setattr(flirter.inputs, key, value)
        assert flirter.cmdline == cmdline
    flirter = fsl.FLIRT(in_file=infile, reference=reffile)
    pth, fname, ext = split_filename(infile)
    flirter.inputs.out_file = ''.join(['foo', ext])
    flirter.inputs.out_matrix_file = ''.join(['bar', ext])
    outs = flirter._list_outputs()
    assert outs['out_file'] == os.path.join(os.getcwd(), flirter.inputs.out_file)
    assert outs['out_matrix_file'] == os.path.join(os.getcwd(), flirter.inputs.out_matrix_file)
    assert not isdefined(flirter.inputs.out_log)