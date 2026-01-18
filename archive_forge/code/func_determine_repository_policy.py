import contextlib
import os
from dulwich.refs import SymrefLoop
from .. import branch as _mod_branch
from .. import errors as brz_errors
from .. import osutils, trace, urlutils
from ..controldir import (BranchReferenceLoop, ControlDir, ControlDirFormat,
from ..transport import (FileExists, NoSuchFile, do_catching_redirections,
from .mapping import decode_git_path, encode_git_path
from .push import GitPushResult
from .transportgit import OBJECTDIR, TransportObjectStore
def determine_repository_policy(self, force_new_repo=False, stack_on=None, stack_on_pwd=None, require_stacking=False):
    """Return an object representing a policy to use.

        This controls whether a new repository is created, and the format of
        that repository, or some existing shared repository used instead.

        If stack_on is supplied, will not seek a containing shared repo.

        :param force_new_repo: If True, require a new repository to be created.
        :param stack_on: If supplied, the location to stack on.  If not
            supplied, a default_stack_on location may be used.
        :param stack_on_pwd: If stack_on is relative, the location it is
            relative to.
        """
    return UseExistingRepository(self.find_repository())