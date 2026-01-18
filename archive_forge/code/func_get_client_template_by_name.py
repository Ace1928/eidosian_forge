from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_client_template_by_name(self, name, realm='master'):
    """ Obtain client template representation by name

        :param name: name of client template to be queried
        :param realm: client template from this realm
        :return: dict of client template representation or None if none matching exist
        """
    result = self.get_client_templates(realm)
    if isinstance(result, list):
        result = [x for x in result if x['name'] == name]
        if len(result) > 0:
            return result[0]
    return None