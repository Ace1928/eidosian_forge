from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_checkpoint(self):
    c = commands.CheckpointCommand()
    self.assertEqual(b'checkpoint', bytes(c))