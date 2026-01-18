import gzip
import re
from dulwich.refs import SymrefLoop
from .. import config, debug, errors, osutils, trace, ui, urlutils
from ..controldir import BranchReferenceLoop
from ..errors import (AlreadyBranchError, BzrError, ConnectionReset,
from ..push import PushResult
from ..revision import NULL_REVISION
from ..revisiontree import RevisionTree
from ..transport import (NoSuchFile, Transport,
from . import is_github_url, lazy_check_versions, user_agent_for_github
import os
import select
import urllib.parse as urlparse
import dulwich
import dulwich.client
from dulwich.errors import GitProtocolError, HangupException
from dulwich.pack import (PACK_SPOOL_FILE_MAX_SIZE, Pack, load_pack_index,
from dulwich.protocol import ZERO_SHA
from dulwich.refs import SYMREF, DictRefsContainer
from dulwich.repo import NotGitRepository
from .branch import (GitBranch, GitBranchFormat, GitBranchPushResult, GitTags,
from .dir import GitControlDirFormat, GitDir
from .errors import GitSmartRemoteNotSupported
from .mapping import encode_git_path, mapping_registry
from .object_store import get_object_store
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_peeled, ref_to_tag_name,
from .repository import GitRepository, GitRepositoryFormat
def get_branch_reference(self, name=None):
    ref = self._get_selected_ref(name)
    try:
        ref_chain, unused_sha = self.get_refs_container().follow(ref)
    except SymrefLoop:
        raise BranchReferenceLoop(self)
    if len(ref_chain) == 1:
        return None
    target_ref = ref_chain[1]
    from .refs import ref_to_branch_name
    try:
        branch_name = ref_to_branch_name(target_ref)
    except ValueError:
        params = {'ref': urlutils.quote(target_ref.decode('utf-8'), '')}
    else:
        if branch_name != '':
            params = {'branch': urlutils.quote(branch_name, '')}
        else:
            params = {}
    return urlutils.join_segment_parameters(self.user_url.rstrip('/'), params)