from __future__ import absolute_import, division, print_function
import json
import os
import uuid
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def get_uri_with_slot_number_query_param(self, uri):
    """Return the URI with proxy slot number added as a query param, if there is one.

        If a proxy slot number is provided, to access it, we must append it as a query parameter.
        This method returns the given URI with the slotnumber query param added, if there is one.
        If there is not a proxy slot number, it just returns the URI as it was passed in.
        """
    if self.proxy_slot_number is not None:
        parsed_url = urlparse(uri)
        return parsed_url._replace(query='slotnumber=' + str(self.proxy_slot_number)).geturl()
    else:
        return uri