from __future__ import absolute_import, division, print_function
from ansible.module_utils.connection import Connection
from ansible.module_utils.six.moves.urllib.parse import quote_plus
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.splunk.es.plugins.module_utils.splunk import (
from ansible_collections.splunk.es.plugins.modules.splunk_data_inputs_monitor import DOCUMENTATION
def search_for_resource_name(self, conn_request, directory_name):
    query_dict = conn_request.get_by_path('{0}/{1}'.format(self.api_object, quote_plus(directory_name)))
    search_result = {}
    if query_dict:
        search_result = self.map_params_to_object(query_dict['entry'][0])
    return search_result