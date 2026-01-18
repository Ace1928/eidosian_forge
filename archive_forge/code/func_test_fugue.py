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
@pytest.mark.parametrize('attr, out_file', [({'save_unmasked_fmap': True, 'fmap_in_file': 'infile', 'mask_file': 'infile', 'output_type': 'NIFTI_GZ'}, 'fmap_out_file'), ({'save_unmasked_shift': True, 'fmap_in_file': 'infile', 'dwell_time': 0.001, 'mask_file': 'infile', 'output_type': 'NIFTI_GZ'}, 'shift_out_file'), ({'in_file': 'infile', 'mask_file': 'infile', 'shift_in_file': 'infile', 'output_type': 'NIFTI_GZ'}, 'unwarped_file')])
def test_fugue(setup_fugue, attr, out_file):
    import os.path as op
    tmpdir, infile = setup_fugue
    fugue = fsl.FUGUE()
    for key, value in attr.items():
        if value == 'infile':
            setattr(fugue.inputs, key, infile)
        else:
            setattr(fugue.inputs, key, value)
    res = fugue.run()
    assert isdefined(getattr(res.outputs, out_file))
    trait_spec = fugue.inputs.trait(out_file)
    out_name = trait_spec.name_template % 'dumbfile'
    out_name += '.nii.gz'
    assert op.basename(getattr(res.outputs, out_file)) == out_name