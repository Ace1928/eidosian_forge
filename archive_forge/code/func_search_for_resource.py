from __future__ import absolute_import, division, print_function
import json
from ansible.plugins.action import ActionBase
from ansible.module_utils.connection import Connection
from ansible.module_utils._text import to_text
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.ibm.qradar.plugins.module_utils.qradar import (
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ibm.qradar.plugins.modules.qradar_analytics_rules import (
def search_for_resource(self, qradar_request, search_for_resource=None):
    """The fn TC of GATHER operation
        :param qradar_request: Qradar connection request
        :param search_for_resource: Module input config with either ID, NAME, or RANGE with field input
        :rtype: A dict
        :returns: dict with module prams transformed having API expected params
        """
    if search_for_resource.get('id'):
        api_obj_url = self.api_object + '/{0}'.format(search_for_resource['id'])
    elif search_for_resource.get('name'):
        api_obj_url = self.api_object + '?filter={0}'.format(quote('name="{0}"'.format(to_text(search_for_resource['name']))))
    elif search_for_resource.get('range'):
        api_obj_url = self.api_object
    if search_for_resource.get('fields'):
        fields = ','.join(search_for_resource['fields'])
        fields_url = '?fields={0}'.format(quote('{0}'.format(fields)))
        api_obj_url += fields_url
    code, rule_source_exists = qradar_request.get(api_obj_url)
    if rule_source_exists and len(rule_source_exists) == 1 and (search_for_resource.get('name') and (not search_for_resource.get('id'))):
        rule_source_exists = rule_source_exists[0]
    return rule_source_exists