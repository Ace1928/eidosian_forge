import sys
from io import BytesIO  # NOQA
import numpy as np
from numpy.testing import measure
from nibabel.volumeutils import array_to_file  # NOQA
from .butils import print_git_title
def bench_array_to_file():
    rng = np.random.RandomState(20111001)
    repeat = 10
    img_shape = (128, 128, 64, 10)
    arr = rng.normal(size=img_shape)
    sys.stdout.flush()
    print_git_title('\nArray to file')
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save float64 to float32', mtime))
    mtime = measure('array_to_file(arr, BytesIO(), np.int16)', repeat)
    print('%30s %6.2f' % ('Save float64 to int16', mtime))
    arr[:, :, :, 1] = np.nan
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save float64 to float32, NaNs', mtime))
    mtime = measure('array_to_file(arr, BytesIO(), np.int16)', repeat)
    print('%30s %6.2f' % ('Save float64 to int16, NaNs', mtime))
    arr[:, :, :, 1] = np.inf
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save float64 to float32, infs', mtime))
    mtime = measure('array_to_file(arr, BytesIO(), np.int16)', repeat)
    print('%30s %6.2f' % ('Save float64 to int16, infs', mtime))
    arr = np.random.random_integers(low=-1000, high=1000, size=img_shape)
    arr = arr.astype(np.int16)
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save Int16 to float32', mtime))
    sys.stdout.flush()