import time
from io import BytesIO
from dulwich import __version__ as dulwich_version
from dulwich.objects import Blob
from .. import __version__ as brz_version
from .. import branch as _mod_branch
from .. import diff as _mod_diff
from .. import errors, osutils
from .. import revision as _mod_revision
from ..merge_directive import BaseMergeDirective
from .mapping import object_mode
from .object_store import get_object_store
def send_git(branch, revision_id, submit_branch, public_branch, no_patch, no_bundle, message, base_revision_id, local_target_branch=None):
    if no_patch:
        raise errors.CommandError('no patch not supported for git-am style patches')
    if no_bundle:
        raise errors.CommandError('no bundle not supported for git-am style patches')
    return GitMergeDirective.from_objects(branch.repository, revision_id, time.time(), osutils.local_time_offset(), submit_branch, public_branch=public_branch, message=message, local_target_branch=local_target_branch)