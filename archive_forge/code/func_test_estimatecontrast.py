import os
import nipype.interfaces.spm.model as spm
import nipype.interfaces.matlab as mlab
def test_estimatecontrast():
    assert spm.EstimateContrast._jobtype == 'stats'
    assert spm.EstimateContrast._jobname == 'con'