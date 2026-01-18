from pathlib import Path
import numpy as np
import scipy.io
import scipy.io.wavfile
from scipy._lib._tmpdirs import tempdir
import scipy.sparse
def test_netcdf_file(self):
    path = Path(__file__).parent / 'data/example_1.nc'
    scipy.io.netcdf_file(path)