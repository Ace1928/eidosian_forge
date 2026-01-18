import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def make_basic_tree(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/a', self._file_content)])
    tree.add('a')
    tree.commit('1')
    return tree