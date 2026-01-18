import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_identity_creation(self):
    from kivy.graphics import LoadIdentity
    mat = LoadIdentity()
    self.assertTrue(mat.stack)