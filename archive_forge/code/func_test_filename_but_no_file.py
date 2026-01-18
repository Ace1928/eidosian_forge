import os
import pickle
import pickletools
import shutil
import tempfile
import unittest
import unittest.mock as mock
import warnings
from traits.api import HasTraits, Int
from traits.testing.optional_dependencies import requires_traitsui, traitsui
def test_filename_but_no_file(self):
    model = Model(count=37)
    filename = os.path.join(self.tmpdir, 'nonexistent.pkl')
    self.assertFalse(os.path.exists(filename))
    with mock.patch.object(self.toolkit, 'view_application'):
        with self.assertWarns(DeprecationWarning):
            model.configure_traits(filename=filename)
    self.assertTrue(os.path.exists(filename))
    with open(filename, 'rb') as pickled_object:
        unpickled = pickle.load(pickled_object)
    self.assertIsInstance(unpickled, Model)
    self.assertEqual(unpickled.count, model.count)