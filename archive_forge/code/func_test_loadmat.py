from pathlib import Path
import numpy as np
import scipy.io
import scipy.io.wavfile
from scipy._lib._tmpdirs import tempdir
import scipy.sparse
def test_loadmat(self):
    with tempdir() as temp_dir:
        path = Path(temp_dir) / 'data.mat'
        scipy.io.savemat(str(path), {'data': self.data})
        mat_contents = scipy.io.loadmat(path)
        assert (mat_contents['data'] == self.data).all()