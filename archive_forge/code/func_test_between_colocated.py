from io import BytesIO
from testtools.matchers import Equals, MatchesAny
from ... import branch, check, controldir, errors, push, tests
from ...branch import BindingUnsupported, Branch
from ...bzr import branch as bzrbranch
from ...bzr import vf_repository
from ...bzr.smart.repository import SmartServerRepositoryGetParentMap
from ...controldir import ControlDir
from ...revision import NULL_REVISION
from .. import test_server
from . import TestCaseWithInterBranch
def test_between_colocated(self):
    """Pushing from one colocated branch to another doesn't change the active branch."""
    source = self.make_from_branch_and_tree('source')
    target = self.make_to_branch('target')
    self.build_tree(['source/a'])
    source.add(['a'])
    revid1 = source.commit('a')
    self.build_tree(['source/b'])
    source.add(['b'])
    revid2 = source.commit('b')
    source_colo = source.controldir.create_branch('colo')
    source_colo.generate_revision_history(revid1)
    try:
        source_colo.push(target)
    except errors.NoRoundtrippingSupport:
        raise tests.TestNotApplicable('push between branches of different format')
    self.assertEqual(source_colo.last_revision(), revid1)
    self.assertEqual(source.last_revision(), revid2)
    self.assertEqual(target.last_revision(), revid1)