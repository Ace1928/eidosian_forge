import unittest
from unittest import mock
from traits.testing.optional_dependencies import requires_traitsui
from traits.util._traitsui_helpers import check_traitsui_major_version
def test_check_version_okay(self):
    with mock.patch('traitsui.__version__', '7.0.0'):
        try:
            check_traitsui_major_version(7)
        except Exception:
            self.fail('Given TraitsUI version is okay, sanity check unexpectedly failed.')