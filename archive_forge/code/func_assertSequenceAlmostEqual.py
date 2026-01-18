import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
def assertSequenceAlmostEqual(self, seq1, seq2, delta=None):
    assert len(seq1) == len(seq2)
    for a, b in zip(seq1, seq2):
        self.assertAlmostEqual(a, b, delta=delta)