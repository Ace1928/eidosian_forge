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
def test_filename_with_existing_file(self):
    stored_model = Model(count=52)
    filename = os.path.join(self.tmpdir, 'model.pkl')
    with open(filename, 'wb') as pickled_object:
        pickle.dump(stored_model, pickled_object)
    model = Model(count=19)
    with mock.patch.object(self.toolkit, 'view_application'):
        with self.assertWarns(DeprecationWarning):
            model.configure_traits(filename=filename)
    self.assertEqual(model.count, 52)