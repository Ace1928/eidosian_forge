import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
def test_file_create_with_meta_block_size_4096(self):
    meta_block_size = 4096
    with File(self.mktemp(), 'w', meta_block_size=meta_block_size, libver='latest') as f:
        f['test'] = 5
        self.assertEqual(f.meta_block_size, meta_block_size)
        self.assertGreaterEqual(f['test'].id.get_offset(), meta_block_size)