from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
@staticmethod
def change_in_initiator_comments(modify, current):
    if 'initiator_objects' not in current:
        return list()
    comments = dict(((item['name'], item['comment']) for item in current['initiator_objects']))

    def has_changed_comment(item):
        return item['name'] in comments and item['comment'] is not None and (item['comment'] != comments[item['name']])
    return [item for item in modify['initiator_objects'] if has_changed_comment(item)]