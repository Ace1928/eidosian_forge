import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def get_manager(self):
    return self.make_branch_and_tree('.').get_shelf_manager()