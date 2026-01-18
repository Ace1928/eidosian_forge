from dulwich.objects import Tag, object_class
from dulwich.refs import (LOCAL_BRANCH_PREFIX, LOCAL_TAG_PREFIX)
from dulwich.repo import RefsContainer
from .. import controldir, errors, osutils
from .. import revision as _mod_revision
def branch_name_to_ref(name):
    """Map a branch name to a ref.

    :param name: Branch name
    :return: ref string
    """
    if name == '':
        return b'HEAD'
    if not name.startswith('refs/'):
        return LOCAL_BRANCH_PREFIX + osutils.safe_utf8(name)
    else:
        return osutils.safe_utf8(name)