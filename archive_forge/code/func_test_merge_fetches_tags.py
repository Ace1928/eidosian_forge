import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_fetches_tags(self):
    """Tags are updated by merge, and revisions named in those tags are
        fetched.
        """
    builder = self.make_branch_builder('source')
    builder.build_commit(message='Rev 1', rev_id=b'rev-1')
    source = builder.get_branch()
    target_bzrdir = source.controldir.sprout('target')
    builder.build_commit(message='Rev 2a', rev_id=b'rev-2a')
    source.tags.set_tag('tag-a', b'rev-2a')
    source.set_last_revision_info(1, b'rev-1')
    source.get_config_stack().set('branch.fetch_tags', True)
    builder.build_commit(message='Rev 2b', rev_id=b'rev-2b')
    self.run_bzr('merge -d target source')
    target = target_bzrdir.open_branch()
    self.assertEqual(b'rev-2a', target.tags.lookup_tag('tag-a'))
    target.repository.get_revision(b'rev-2a')