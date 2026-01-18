from ansible.module_utils.network.aos.aos import (check_aos_version, get_aos_session, find_collection_item,
from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.network.plugins.module_utils.version import LooseVersion
from ansible.module_utils._text import to_native
def get_aos_session(module, auth):
    """
    Resume an existing session and return an AOS object.

    Args:
        auth (dict): An AOS session as obtained by aos_login module blocks::

            dict( token=<token>,
                  server=<ip>,
                  port=<port>
                )

    Return:
        Aos object
    """
    check_aos_version(module)
    aos = Session()
    aos.session = auth
    return aos