import os
import nipype.interfaces.spm.model as spm
import nipype.interfaces.matlab as mlab
def test_level1design():
    assert spm.Level1Design._jobtype == 'stats'
    assert spm.Level1Design._jobname == 'fmri_spec'