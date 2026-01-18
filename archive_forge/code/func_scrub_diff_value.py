from __future__ import absolute_import, division, print_function
import json
from collections import defaultdict
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import iteritems
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.urllib.parse import quote as urlquote
def scrub_diff_value(self, value):
    """
        Scrub the 'command_id' key from the returned data.

        The command api returns the command_id, rendering the diff useless

        Parameters:
            value: type dict, the dict to remove the command_id key from

        Returns:
            the dict value without the key command_id
        """
    if isinstance(value, dict):
        for k, v in iteritems(value.copy()):
            if isinstance(value[k], dict):
                value[k].pop('command_id', None)
    return value