from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_subgroup_by_chain(self, name_chain, realm='master'):
    """ Access a subgroup API object by walking down a given name/id chain.

        Groups can be given either as by name or by ID, the first element
        must either be a toplvl group or given as ID, all parents must exist.

        If the group cannot be found, None is returned.
        :param name_chain: Topdown ordered list of subgroup parent (ids or names) + its own name at the end
        :param realm: Realm in which the group resides; default 'master'
        """
    cp = name_chain[0]
    cp, is_id = self._get_normed_group_parent(cp)
    if is_id:
        tmp = self.get_group_by_groupid(cp, realm=realm)
    else:
        tmp = self.get_group_by_name(cp, realm=realm)
    if not tmp:
        return None
    for p in name_chain[1:]:
        for sg in tmp['subGroups']:
            pv, is_id = self._get_normed_group_parent(p)
            if is_id:
                cmpkey = 'id'
            else:
                cmpkey = 'name'
            if pv == sg[cmpkey]:
                tmp = sg
                break
        if not tmp:
            return None
    return tmp