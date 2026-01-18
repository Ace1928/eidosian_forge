import numpy as np
import numpy.testing as npt
from .. import rapidart as ra
from ...interfaces.base import Bunch
def test_ad_init():
    ad = ra.ArtifactDetect(use_differences=[True, False])
    assert ad.inputs.use_differences[0]
    assert not ad.inputs.use_differences[1]