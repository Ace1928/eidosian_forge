import sys
import breezy
import breezy.errors as errors
import breezy.gpg
from breezy.bzr.inventory import Inventory
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.per_interrepository import TestCaseWithInterRepository
from breezy.workingtree import WorkingTree
def test_fetch_fetches_signatures_too(self):
    if not self.repository_format.supports_revision_signatures:
        raise TestNotApplicable('from repository does not support signatures')
    if not self.repository_format_to.supports_revision_signatures:
        raise TestNotApplicable('to repository does not support signatures')
    tree_a = WorkingTree.open('a')
    with tree_a.branch.repository.lock_write(), WriteGroup(tree_a.branch.repository):
        tree_a.branch.repository.sign_revision(self.rev2, breezy.gpg.LoopbackGPGStrategy(None))
    from_repo = self.controldir.open_repository()
    from_signature = from_repo.get_signature_text(self.rev2)
    to_repo = self.make_to_repository('target')
    try:
        to_repo.fetch(from_repo)
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('interrepo does not support roundtripping')
    to_signature = to_repo.get_signature_text(self.rev2)
    self.assertEqual(from_signature, to_signature)