import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def load_plugin_tests(self, loader):
    """Return the adapted plugin's test suite.

        Args:
          loader: The custom loader that should be used to load additional
            tests.
        """
    if getattr(self.module, 'load_tests', None) is not None:
        return loader.loadTestsFromModule(self.module)
    else:
        return None