import argparse
import sys
from unittest import mock
from osc_lib.tests import utils as test_utils
from osc_lib.utils import tags
def help_enhancer(_h):
    """A simple helper to validate the ``enhance_help`` kwarg."""
    return ''.join(reversed(_h))