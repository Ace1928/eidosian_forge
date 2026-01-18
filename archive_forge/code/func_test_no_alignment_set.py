import h5py
from .common import TestCase
def test_no_alignment_set(self):
    fname = self.mktemp()
    shape = (881,)
    with h5py.File(fname, 'w') as h5file:
        for i in range(1000):
            dataset = h5file.create_dataset(dataset_name(i), shape, dtype='uint8')
            dataset[...] = i
            if not is_aligned(dataset):
                break
        else:
            raise RuntimeError('Data was all found to be aligned to 4096')