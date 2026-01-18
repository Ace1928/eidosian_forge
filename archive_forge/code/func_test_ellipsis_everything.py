from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_ellipsis_everything(self):
    """A simple ellipsis matches everything."""
    self.run_script('\n        $ echo foo\n        ...\n        ')