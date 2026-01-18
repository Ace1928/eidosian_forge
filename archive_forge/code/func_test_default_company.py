import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_default_company(self):
    """
        default company

        """
    self.assertEqual(self.ETSConfig.company, 'Enthought')