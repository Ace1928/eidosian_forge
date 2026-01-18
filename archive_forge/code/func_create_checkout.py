import contextlib
from collections import defaultdict
from functools import partial
from io import BytesIO
from typing import Dict, Optional, Set, Tuple
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.objects import ZERO_SHA, NotCommitError
from dulwich.repo import check_ref_format
from .. import branch, config, controldir, errors, lock
from .. import repository as _mod_repository
from .. import revision, trace, transport, urlutils
from ..foreign import ForeignBranch
from ..revision import NULL_REVISION
from ..tag import InterTags, TagConflict, Tags, TagSelector, TagUpdates
from ..trace import is_quiet, mutter, warning
from .errors import NoPushSupport
from .mapping import decode_git_path, encode_git_path
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_tag, ref_to_branch_name,
from .unpeel_map import UnpeelMap
from .urls import bzr_url_to_git_url, git_url_to_bzr_url
def create_checkout(self, to_location, revision_id=None, lightweight=False, accelerator_tree=None, hardlink=False):
    t = transport.get_transport(to_location)
    t.ensure_base()
    format = self._get_checkout_format(lightweight=lightweight)
    checkout = format.initialize_on_transport(t)
    if lightweight:
        from_branch = checkout.set_branch_reference(target_branch=self)
    else:
        policy = checkout.determine_repository_policy()
        policy.acquire_repository()
        checkout_branch = checkout.create_branch()
        checkout_branch.bind(self)
        checkout_branch.pull(self, stop_revision=revision_id)
        from_branch = None
    return checkout.create_workingtree(revision_id, from_branch=from_branch, hardlink=hardlink)