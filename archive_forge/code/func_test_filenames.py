from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_filenames(self):
    vf = multiparent.MultiVersionedFile('foop')
    vf.add_version(b'a\nb\nc\nd'.splitlines(True), b'a', [])
    self.assertPathExists('foop.mpknit')
    self.assertPathDoesNotExist('foop.mpidx')
    vf.save()
    self.assertPathExists('foop.mpidx')
    vf.destroy()
    self.assertPathDoesNotExist('foop.mpknit')
    self.assertPathDoesNotExist('foop.mpidx')