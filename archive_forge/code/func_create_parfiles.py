import os
import numpy as np
import nibabel as nb
import pytest
import nipype.interfaces.fsl.utils as fsl
from nipype.interfaces.fsl import no_fsl, Info
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
def create_parfiles():
    np.savetxt('a.par', np.random.rand(6, 3))
    np.savetxt('b.par', np.random.rand(6, 3))
    return ['a.par', 'b.par']