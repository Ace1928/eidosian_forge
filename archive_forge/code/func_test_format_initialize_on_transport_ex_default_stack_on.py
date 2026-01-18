import errno
from stat import S_ISDIR
import breezy.branch
from breezy import controldir, errors, repository
from breezy import revision as _mod_revision
from breezy import transport, workingtree
from breezy.bzr import bzrdir
from breezy.bzr.remote import RemoteBzrDirFormat
from breezy.bzr.tests.per_bzrdir import TestCaseWithBzrDir
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.transport import FileExists
from breezy.transport.local import LocalTransport
def test_format_initialize_on_transport_ex_default_stack_on(self):
    balloon = self.make_controldir('balloon')
    if isinstance(balloon._format, bzrdir.BzrDirMetaFormat1):
        stack_on = self.make_branch('stack-on', format='1.9')
    else:
        stack_on = self.make_branch('stack-on')
    if not stack_on.repository._format.supports_nesting_repositories:
        raise TestNotApplicable('requires nesting repositories')
    config = self.make_controldir('.').get_config()
    try:
        config.set_default_stack_on('stack-on')
    except errors.BzrError:
        raise TestNotApplicable('Only relevant for stackable formats.')
    t = self.get_transport('stacked')
    repo_fmt = controldir.format_registry.make_controldir('1.9')
    repo_name = repo_fmt.repository_format.network_name()
    repo, control = self.assertInitializeEx(t, need_meta=True, repo_format_name=repo_name, stacked_on=None)
    self.assertLength(1, repo._fallback_repositories)
    fallback_repo = repo._fallback_repositories[0]
    self.assertEqual(stack_on.base, fallback_repo.controldir.root_transport.base)
    new_branch = control.create_branch()
    self.assertTrue(new_branch._format.supports_stacking())