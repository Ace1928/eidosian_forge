import os
import numpy as np
from ...testing import utils
from .. import nilearn as iface
from ...pipeline import engine as pe
import pytest
import numpy.testing as npt
def assert_expected_output(self, labels, wanted):
    with open(self.filenames['out_file'], 'r') as output:
        got = [line.split() for line in output]
        labels_got = got.pop(0)
        assert labels_got == labels
        assert len(got) == self.fake_fmri_data.shape[3], 'num rows and num volumes'
        got = [[float(num) for num in row] for row in got]
        for i, time in enumerate(got):
            assert len(labels) == len(time)
            for j, segment in enumerate(time):
                npt.assert_almost_equal(segment, wanted[i][j], decimal=1)