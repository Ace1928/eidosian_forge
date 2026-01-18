from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def check_subscr(self):
    """Check the subscription and refresh ``self.attrs`` subscription attribute.

        Returns:
            True if the subscription with ``self.name`` exists, False otherwise.
        """
    subscr_info = self.__get_general_subscr_info()
    if not subscr_info:
        self.attrs = deepcopy(self.empty_attrs)
        return False
    self.attrs['owner'] = subscr_info.get('rolname')
    self.attrs['enabled'] = subscr_info.get('subenabled')
    self.attrs['synccommit'] = subscr_info.get('subenabled')
    self.attrs['slotname'] = subscr_info.get('subslotname')
    self.attrs['publications'] = subscr_info.get('subpublications')
    if subscr_info.get('comment') is not None:
        self.attrs['comment'] = subscr_info.get('comment')
    else:
        self.attrs['comment'] = ''
    if subscr_info.get('subconninfo'):
        for param in subscr_info['subconninfo'].split(' '):
            tmp = param.split('=')
            try:
                self.attrs['conninfo'][tmp[0]] = int(tmp[1])
            except ValueError:
                self.attrs['conninfo'][tmp[0]] = tmp[1]
    return True