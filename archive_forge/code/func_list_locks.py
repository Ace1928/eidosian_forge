from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils.common.dict_transformations import _camel_to_snake
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
def list_locks(self, url):
    try:
        resp = self._mgmt_client.query(url=url, method='GET', query_parameters=self._query_parameters, header_parameters=self._header_parameters, body=None, expected_status_codes=[200], polling_timeout=None, polling_interval=None)
        return json.loads(resp.body())
    except Exception as exc:
        self.fail('Error when finding locks {0}: {1}'.format(url, exc.message))