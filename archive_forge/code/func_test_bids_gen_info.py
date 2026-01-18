from copy import deepcopy
import os
from nibabel import Nifti1Image
import numpy as np
import pytest
import numpy.testing as npt
from nipype.testing import example_data
from nipype.interfaces.base import Bunch, TraitError
from nipype.algorithms.modelgen import (
def test_bids_gen_info():
    fname = example_data('events.tsv')
    res = bids_gen_info([fname])
    assert res[0].onsets == [[183.75, 313.75, 483.75, 633.75, 783.75, 933.75, 1083.75, 1233.75]]
    assert res[0].durations == [[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]]
    assert res[0].amplitudes == [[1, 1, 1, 1, 1, 1, 1, 1]]
    assert res[0].conditions == ['ev0']