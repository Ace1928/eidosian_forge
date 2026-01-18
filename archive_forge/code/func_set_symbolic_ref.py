from dulwich.objects import Tag, object_class
from dulwich.refs import (LOCAL_BRANCH_PREFIX, LOCAL_TAG_PREFIX)
from dulwich.repo import RefsContainer
from .. import controldir, errors, osutils
from .. import revision as _mod_revision
def set_symbolic_ref(self, name, other):
    if name == b'HEAD':
        pass
    else:
        raise NotImplementedError('Symbolic references not supported for anything other than HEAD')