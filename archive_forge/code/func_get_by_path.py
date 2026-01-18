from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlencode
def get_by_path(self, rest_path):
    """
        GET attributes of a monitor by rest path
        """
    return self.get('/{0}?output_mode=json'.format(rest_path))