import unittest
from subunit.progress_model import ProgressModel
def test_adjust_width_adjusts_substack(self):
    progress = ProgressModel()
    progress.adjust_width(3)
    progress.advance()
    progress.push()
    progress.adjust_width(2)
    progress.advance()
    self.assertProgressSummary(3, 6, progress)