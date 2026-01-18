import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_immediate(self):
    gdb = Recognizer(db=[self.Tinvar, self.Ninvar])
    r = gdb.recognize([Ncandidate], max_gpf=0)
    self.assertEqual(r._match_ops, 4)
    self.assertEqual(r._completed, 2)
    self.assertEqual(r.progress, 1)
    self.assertTrue(r.best['score'] > 0.94 and r.best['score'] < 0.95)