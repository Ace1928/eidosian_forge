import os
import sys
from textwrap import dedent
from twisted.persisted import sob
from twisted.persisted.styles import Ephemeral
from twisted.python import components
from twisted.trial import unittest
def testTypeGuesser(self):
    self.assertRaises(KeyError, sob.guessType, 'file.blah')
    self.assertEqual('python', sob.guessType('file.py'))
    self.assertEqual('python', sob.guessType('file.tac'))
    self.assertEqual('python', sob.guessType('file.etac'))
    self.assertEqual('pickle', sob.guessType('file.tap'))
    self.assertEqual('pickle', sob.guessType('file.etap'))
    self.assertEqual('source', sob.guessType('file.tas'))
    self.assertEqual('source', sob.guessType('file.etas'))