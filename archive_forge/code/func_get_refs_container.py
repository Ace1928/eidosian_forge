from dulwich.objects import Tag, object_class
from dulwich.refs import (LOCAL_BRANCH_PREFIX, LOCAL_TAG_PREFIX)
from dulwich.repo import RefsContainer
from .. import controldir, errors, osutils
from .. import revision as _mod_revision
def get_refs_container(controldir, object_store):
    fn = getattr(controldir, 'get_refs_container', None)
    if fn is not None:
        return fn()
    return BazaarRefsContainer(controldir, object_store)