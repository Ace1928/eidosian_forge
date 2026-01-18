import unittest
from subunit.progress_model import ProgressModel
def test_adjust_width(self):
    progress = ProgressModel()
    progress.adjust_width(10)
    self.assertProgressSummary(0, 10, progress)
    progress.adjust_width(-10)
    self.assertProgressSummary(0, 0, progress)