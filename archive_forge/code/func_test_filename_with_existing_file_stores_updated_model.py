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
def test_filename_with_existing_file_stores_updated_model(self):
    stored_model = Model(count=52)
    filename = os.path.join(self.tmpdir, 'model.pkl')
    with open(filename, 'wb') as pickled_object:
        pickle.dump(stored_model, pickled_object)

    def modify_model(*args, **kwargs):
        model.count = 23
        return mock.DEFAULT
    model = Model(count=19)
    with mock.patch.object(self.toolkit, 'view_application') as mock_view:
        mock_view.side_effect = modify_model
        with self.assertWarns(DeprecationWarning):
            model.configure_traits(filename=filename)
    self.assertEqual(model.count, 23)
    with open(filename, 'rb') as pickled_object:
        unpickled = pickle.load(pickled_object)
    self.assertIsInstance(unpickled, Model)
    self.assertEqual(unpickled.count, model.count)