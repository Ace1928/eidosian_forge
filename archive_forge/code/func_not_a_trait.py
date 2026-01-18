import contextlib
import io
import os
import shutil
import tempfile
import textwrap
import tokenize
import unittest
import unittest.mock as mock
from traits.api import Bool, HasTraits, Int, Property
from traits.testing.optional_dependencies import sphinx, requires_sphinx
@property
def not_a_trait(self):
    """
        I'm a regular property, not a trait.
        """