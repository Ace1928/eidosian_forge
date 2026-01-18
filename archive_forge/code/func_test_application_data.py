import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_application_data(self):
    """
        application data

        """
    dirname = self.ETSConfig.application_data
    self.assertEqual(os.path.exists(dirname), True)
    self.assertEqual(os.path.isdir(dirname), True)