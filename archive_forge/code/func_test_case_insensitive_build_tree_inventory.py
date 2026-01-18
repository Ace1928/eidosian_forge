import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_case_insensitive_build_tree_inventory(self):
    if features.CaseInsensitiveFilesystemFeature.available() or features.CaseInsCasePresFilenameFeature.available():
        raise UnavailableFeature('Fully case sensitive filesystem')
    source = self.make_branch_and_tree('source')
    self.build_tree(['source/file', 'source/FILE'])
    source.add(['file', 'FILE'], ids=[b'lower-id', b'upper-id'])
    source.commit('added files')
    target = self.make_branch_and_tree('target')
    target.case_sensitive = False
    build_tree(source.basis_tree(), target, source, delta_from_tree=True)
    self.assertEqual('file.moved', target.id2path(b'lower-id'))
    self.assertEqual('FILE', target.id2path(b'upper-id'))