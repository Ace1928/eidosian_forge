import logging
import unittest
from collections import namedtuple
from gensim.topic_coherence import direct_confirmation_measure
from gensim.topic_coherence import text_analysis
def test_log_ratio_measure(self):
    """Test log_ratio_measure()"""
    obtained = direct_confirmation_measure.log_ratio_measure(self.segmentation, self.accumulator)[0]
    expected = -0.182321557
    self.assertAlmostEqual(expected, obtained)
    mean, std = direct_confirmation_measure.log_ratio_measure(self.segmentation, self.accumulator, with_std=True)[0]
    self.assertAlmostEqual(expected, mean)
    self.assertEqual(0.0, std)