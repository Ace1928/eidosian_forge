import os
import time
import shutil
import signal
import subprocess
from subprocess import CalledProcessError
from tempfile import mkdtemp
from ..utils.misc import package_check
import numpy as np
import nibabel as nb
def save_toy_nii(ndarray, filename):
    toy = nb.Nifti1Image(ndarray, np.eye(4))
    nb.nifti1.save(toy, filename)
    return filename