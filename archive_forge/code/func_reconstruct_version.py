from unittest import TestCase
import patiencediff
from .. import multiparent, tests
@staticmethod
def reconstruct_version(vf, revision_id):
    reconstructor = multiparent._Reconstructor(vf, vf._lines, vf._parents)
    lines = []
    reconstructor.reconstruct_version(lines, revision_id)
    return lines