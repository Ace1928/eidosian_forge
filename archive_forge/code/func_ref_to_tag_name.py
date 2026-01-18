from dulwich.objects import Tag, object_class
from dulwich.refs import (LOCAL_BRANCH_PREFIX, LOCAL_TAG_PREFIX)
from dulwich.repo import RefsContainer
from .. import controldir, errors, osutils
from .. import revision as _mod_revision
def ref_to_tag_name(ref):
    if ref.startswith(LOCAL_TAG_PREFIX):
        return ref[len(LOCAL_TAG_PREFIX):].decode('utf-8')
    raise ValueError('unable to map ref %s back to tag name' % ref)