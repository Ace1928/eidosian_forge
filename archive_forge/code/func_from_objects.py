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
@classmethod
def from_objects(cls, repository, revision_id, time, timezone, target_branch, local_target_branch=None, public_branch=None, message=None):
    patches = []
    submit_branch = _mod_branch.Branch.open(target_branch)
    with submit_branch.lock_read():
        submit_revision_id = submit_branch.last_revision()
        repository.fetch(submit_branch.repository, submit_revision_id)
        graph = repository.get_graph()
        todo = graph.find_difference(submit_revision_id, revision_id)[1]
        total = len(todo)
        for i, revid in enumerate(graph.iter_topo_order(todo)):
            patches.append(cls._generate_commit(repository, revid, i + 1, total))
    return cls(revision_id, None, time, timezone, target_branch=target_branch, source_branch=public_branch, message=message, patches=patches)