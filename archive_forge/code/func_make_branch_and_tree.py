from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def make_branch_and_tree(self, path, format='git'):
    return super().make_branch_and_tree(path, format=format)