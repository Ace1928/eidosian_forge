import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def test_clone_stacking_policy_upgrades(self):
    """Cloning an unstackable branch format to somewhere with a default
        stack-on branch upgrades branch and repo to match the target and honour
        the policy.
        """
    try:
        repo = self.make_repository('repo', shared=True)
    except errors.IncompatibleFormat:
        raise tests.TestNotApplicable('Cannot make a shared repository')
    if repo.controldir._format.fixed_components:
        self.knownFailure('pre metadir branches do not upgrade on push with stacking policy')
    if isinstance(repo._format, knitpack_repo.RepositoryFormatKnitPack5RichRootBroken):
        raise tests.TestNotApplicable('unsupported format')
    bzrdir_format = self.repository_format._matchingcontroldir
    transport = self.get_transport('repo/branch')
    transport.mkdir('.')
    target_bzrdir = bzrdir_format.initialize_on_transport(transport)
    branch = _mod_bzrbranch.BzrBranchFormat6().initialize(target_bzrdir)
    if isinstance(repo, remote.RemoteRepository):
        repo._ensure_real()
        info_repo = repo._real_repository
    else:
        info_repo = repo
    format_description = info.describe_format(info_repo.controldir, info_repo, None, None)
    formats = format_description.split(' or ')
    stack_on_format = formats[0]
    if stack_on_format in ['pack-0.92', 'dirstate', 'metaweave']:
        stack_on_format = '1.9'
    elif stack_on_format in ['dirstate-with-subtree', 'rich-root', 'rich-root-pack', 'pack-0.92-subtree']:
        stack_on_format = '1.9-rich-root'
    stack_on = self.make_branch('stack-on-me', format=stack_on_format)
    self.make_controldir('.').get_config().set_default_stack_on('stack-on-me')
    target = branch.controldir.clone(self.get_url('target'))
    self.assertTrue(target.open_branch()._format.supports_stacking())
    if isinstance(repo, remote.RemoteRepository):
        repo._ensure_real()
        repo = repo._real_repository
    target_repo = target.open_repository()
    if isinstance(target_repo, remote.RemoteRepository):
        target_repo._ensure_real()
        target_repo = target_repo._real_repository
    if repo._format.supports_external_lookups:
        self.assertEqual(repo._format, target_repo._format)
    else:
        self.assertEqual(stack_on.repository._format, target_repo._format)