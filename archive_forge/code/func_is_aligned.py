import h5py
from .common import TestCase
def is_aligned(dataset, offset=4096):
    return dataset.id.get_offset() % offset == 0