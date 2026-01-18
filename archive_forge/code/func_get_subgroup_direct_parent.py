from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_subgroup_direct_parent(self, parents, realm='master', children_to_resolve=None):
    """ Get keycloak direct parent group API object for a given chain of parents.

        To successfully work the API for subgroups we actually don't need
        to "walk the whole tree" for nested groups but only need to know
        the ID for the direct predecessor of current subgroup. This
        method will guarantee us this information getting there with
        as minimal work as possible.

        Note that given parent list can and might be incomplete at the
        upper levels as long as it starts with an ID instead of a name

        If the group does not exist, None is returned.
        :param parents: Topdown ordered list of subgroup parents
        :param realm: Realm in which the group resides; default 'master'
        """
    if children_to_resolve is None:
        parents = list(reversed(parents))
        children_to_resolve = []
    if not parents:
        return self.get_subgroup_by_chain(list(reversed(children_to_resolve)), realm=realm)
    cp = parents[0]
    unused, is_id = self._get_normed_group_parent(cp)
    if is_id:
        return self.get_subgroup_by_chain([cp] + list(reversed(children_to_resolve)), realm=realm)
    else:
        children_to_resolve.append(cp)
        return self.get_subgroup_direct_parent(parents[1:], realm=realm, children_to_resolve=children_to_resolve)