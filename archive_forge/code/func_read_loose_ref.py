from dulwich.objects import Tag, object_class
from dulwich.refs import (LOCAL_BRANCH_PREFIX, LOCAL_TAG_PREFIX)
from dulwich.repo import RefsContainer
from .. import controldir, errors, osutils
from .. import revision as _mod_revision
def read_loose_ref(self, ref):
    try:
        branch_name = ref_to_branch_name(ref)
    except ValueError:
        tag_name = ref_to_tag_name(ref)
        revid = self._get_revid_by_tag_name(tag_name)
    else:
        revid = self._get_revid_by_branch_name(branch_name)
    if revid == _mod_revision.NULL_REVISION:
        return None
    with self.object_store.lock_read():
        return self.object_store._lookup_revision_sha1(revid)