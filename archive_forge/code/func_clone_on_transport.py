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
def clone_on_transport(self, transport, revision_id=None, force_new_repo=False, preserve_stacking=False, stacked_on=None, create_prefix=False, use_existing_dir=True, no_tree=False, tag_selector=None):
    """See ControlDir.clone_on_transport."""
    from ..repository import InterRepository
    from ..transport.local import LocalTransport
    from .mapping import default_mapping
    from .refs import is_peeled
    if no_tree:
        format = BareLocalGitControlDirFormat()
    else:
        format = LocalGitControlDirFormat()
    if stacked_on is not None:
        raise _mod_branch.UnstackableBranchFormat(format, self.user_url)
    target_repo, target_controldir, stacking, repo_policy = format.initialize_on_transport_ex(transport, use_existing_dir=use_existing_dir, create_prefix=create_prefix, force_new_repo=force_new_repo)
    target_repo = target_controldir.find_repository()
    target_git_repo = target_repo._git
    source_repo = self.find_repository()
    interrepo = InterRepository.get(source_repo, target_repo)
    if revision_id is not None:
        determine_wants = interrepo.get_determine_wants_revids([revision_id], include_tags=True, tag_selector=tag_selector)
    else:
        determine_wants = interrepo.determine_wants_all
    pack_hint, _, refs = interrepo.fetch_objects(determine_wants, mapping=default_mapping)
    for name, val in refs.items():
        if is_peeled(name):
            continue
        if val in target_git_repo.object_store:
            target_git_repo.refs[name] = val
    result_dir = LocalGitDir(transport, target_git_repo, format)
    result_branch = result_dir.open_branch()
    try:
        parent = self.open_branch().get_parent()
    except brz_errors.InaccessibleParent:
        pass
    else:
        if parent:
            result_branch.set_parent(parent)
    if revision_id is not None:
        result_branch.set_last_revision(revision_id)
    if not no_tree and isinstance(result_dir.root_transport, LocalTransport):
        if result_dir.open_repository().make_working_trees():
            try:
                local_wt = self.open_workingtree()
            except brz_errors.NoWorkingTree:
                pass
            except brz_errors.NotLocalUrl:
                result_dir.create_workingtree(revision_id=revision_id)
            else:
                local_wt.clone(result_dir, revision_id=revision_id)
    return result_dir