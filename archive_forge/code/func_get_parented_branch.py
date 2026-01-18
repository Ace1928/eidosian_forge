import contextlib
from breezy import branch as _mod_branch
from breezy import config, controldir
from breezy import delta as _mod_delta
from breezy import (errors, lock, merge, osutils, repository, revision, shelf,
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.tests.http_server import HttpServer
from breezy.transport import memory
def get_parented_branch(self):
    wt_a = self.make_branch_and_tree('a')
    self.build_tree(['a/one'])
    wt_a.add(['one'])
    rev1 = wt_a.commit('commit one')
    branch_b = wt_a.branch.controldir.sprout('b', revision_id=rev1).open_branch()
    self.assertEqual(urlutils.strip_segment_parameters(wt_a.branch.user_url), urlutils.strip_segment_parameters(branch_b.get_parent()))
    return branch_b