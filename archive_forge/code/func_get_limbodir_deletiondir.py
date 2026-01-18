import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def get_limbodir_deletiondir(self, wt):
    transform = wt.transform()
    limbodir = transform._limbodir
    deletiondir = transform._deletiondir
    transform.finalize()
    return (limbodir, deletiondir)