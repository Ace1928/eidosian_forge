from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def add_version(self, vf, text, version_id, parent_ids):
    vf.add_version([bytes([t]) + b'\n' for t in bytearray(text)], version_id, parent_ids)