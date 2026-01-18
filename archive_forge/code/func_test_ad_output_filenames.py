import numpy as np
import numpy.testing as npt
from .. import rapidart as ra
from ...interfaces.base import Bunch
def test_ad_output_filenames():
    ad = ra.ArtifactDetect()
    outputdir = '/tmp'
    f = 'motion.nii'
    outlierfile, intensityfile, statsfile, normfile, plotfile, displacementfile, maskfile = ad._get_output_filenames(f, outputdir)
    assert outlierfile == '/tmp/art.motion_outliers.txt'
    assert intensityfile == '/tmp/global_intensity.motion.txt'
    assert statsfile == '/tmp/stats.motion.txt'
    assert normfile == '/tmp/norm.motion.txt'
    assert plotfile == '/tmp/plot.motion.png'
    assert displacementfile == '/tmp/disp.motion.nii'
    assert maskfile == '/tmp/mask.motion.nii'