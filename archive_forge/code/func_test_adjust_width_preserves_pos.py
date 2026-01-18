import unittest
from subunit.progress_model import ProgressModel
def test_adjust_width_preserves_pos(self):
    progress = ProgressModel()
    progress.advance()
    progress.adjust_width(10)
    self.assertProgressSummary(1, 10, progress)
    progress.adjust_width(-10)
    self.assertProgressSummary(1, 0, progress)