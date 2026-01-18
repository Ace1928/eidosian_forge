import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_default_backend_for_qt5_toolkit(self):
    self.ETSConfig.toolkit = 'qt'
    self.assertEqual(self.ETSConfig.kiva_backend, 'image')