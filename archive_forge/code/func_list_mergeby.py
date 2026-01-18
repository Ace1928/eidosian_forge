from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleFilterError
from ansible.module_utils.six import string_types
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.utils.vars import merge_hash
from collections import defaultdict
from operator import itemgetter
def list_mergeby(x, y, index, recursive=False, list_merge='replace'):
    """ Merge 2 lists by attribute 'index'. The function merge_hash from ansible.utils.vars is used.
        This function is used by the function lists_mergeby.
    """
    d = defaultdict(dict)
    for l in (x, y):
        for elem in l:
            if not isinstance(elem, Mapping):
                msg = 'Elements of list arguments for lists_mergeby must be dictionaries. %s is %s'
                raise AnsibleFilterError(msg % (elem, type(elem)))
            if index in elem.keys():
                d[elem[index]].update(merge_hash(d[elem[index]], elem, recursive, list_merge))
    return sorted(d.values(), key=itemgetter(index))