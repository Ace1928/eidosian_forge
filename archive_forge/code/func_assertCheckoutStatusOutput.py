import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def assertCheckoutStatusOutput(self, command_string, lco_tree, shared_repo=None, repo_branch=None, tree_locked=False, branch_locked=False, repo_locked=False, verbose=False, light_checkout=True, checkout_root=None):
    """Check the output of info in a checkout.

        This is not quite a mirror of the info code: rather than using the
        tree being examined to predict output, it uses a bunch of flags which
        allow us, the test writers, to document what *should* be present in
        the output. Removing this separation would remove the value of the
        tests.

        :param path: the path to the light checkout.
        :param lco_tree: the tree object for the light checkout.
        :param shared_repo: A shared repository is in use, expect that in
            the output.
        :param repo_branch: A branch in a shared repository for non light
            checkouts.
        :param tree_locked: If true, expect the tree to be locked.
        :param branch_locked: If true, expect the branch to be locked.
        :param repo_locked: If true, expect the repository to be locked.
            Note that the lco_tree.branch.repository is inspected, and if is not
            actually locked then this parameter is overridden. This is because
            pack repositories do not have any public API for obtaining an
            exclusive repository wide lock.
        :param verbose: verbosity level: 2 or higher to show committers
        """

    def friendly_location(url):
        path = urlutils.unescape_for_display(url, 'ascii')
        try:
            return osutils.relpath(osutils.getcwd(), path)
        except errors.PathNotChild:
            return path
    if tree_locked:
        self.expectFailure('OS locks are exclusive for different processes (Bug #174055)', self.run_brz_subprocess, 'info ' + command_string)
    out, err = self.run_bzr('info %s' % command_string)
    description = {(True, True): 'Lightweight checkout', (True, False): 'Repository checkout', (False, True): 'Lightweight checkout', (False, False): 'Checkout'}[shared_repo is not None, light_checkout]
    format = {True: self._repo_strings, False: 'unnamed'}[light_checkout]
    if repo_locked:
        repo_locked = lco_tree.branch.repository.get_physical_lock_status()
    if repo_locked or branch_locked or tree_locked:

        def locked_message(a_bool):
            if a_bool:
                return 'locked'
            else:
                return 'unlocked'
        expected_lock_output = '\nLock status:\n  working tree: %s\n        branch: %s\n    repository: %s\n' % (locked_message(tree_locked), locked_message(branch_locked), locked_message(repo_locked))
    else:
        expected_lock_output = ''
    tree_data = ''
    extra_space = ''
    if light_checkout:
        tree_data = '  light checkout root: %s\n' % friendly_location(lco_tree.controldir.root_transport.base)
        extra_space = ' '
    if lco_tree.branch.get_bound_location() is not None:
        tree_data += '{}       checkout root: {}\n'.format(extra_space, friendly_location(lco_tree.branch.controldir.root_transport.base))
    if shared_repo is not None:
        branch_data = '   checkout of branch: %s\n    shared repository: %s\n' % (friendly_location(repo_branch.controldir.root_transport.base), friendly_location(shared_repo.controldir.root_transport.base))
    elif repo_branch is not None:
        branch_data = '%s  checkout of branch: %s\n' % (extra_space, friendly_location(repo_branch.controldir.root_transport.base))
    else:
        branch_data = '   checkout of branch: %s\n' % lco_tree.branch.controldir.root_transport.base
    if verbose >= 2:
        verbose_info = '         0 committers\n'
    else:
        verbose_info = ''
    self.assertEqualDiff('{} (format: {})\nLocation:\n{}{}\nFormat:\n       control: Meta directory format 1\n  working tree: {}\n        branch: {}\n    repository: {}\n{}\nControl directory:\n         1 branches\n\nIn the working tree:\n         0 unchanged\n         0 modified\n         0 added\n         0 removed\n         0 renamed\n         0 copied\n         0 unknown\n         0 ignored\n         0 versioned subdirectories\n\nBranch history:\n         0 revisions\n{}\nRepository:\n         0 revisions\n'.format(description, format, tree_data, branch_data, lco_tree._format.get_format_description(), lco_tree.branch._format.get_format_description(), lco_tree.branch.repository._format.get_format_description(), expected_lock_output, verbose_info), out)
    self.assertEqual('', err)