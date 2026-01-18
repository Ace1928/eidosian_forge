from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def get_xenserver_version(module):
    """Returns XenServer version.

    Args:
        module: Reference to Ansible module object.

    Returns:
        list: Element [0] is major version. Element [1] is minor version.
        Element [2] is update number.
    """
    xapi_session = XAPI.connect(module)
    host_ref = xapi_session.xenapi.session.get_this_host(xapi_session._session)
    try:
        xenserver_version = [int(version_number) for version_number in xapi_session.xenapi.host.get_software_version(host_ref)['product_version'].split('.')]
    except ValueError:
        xenserver_version = [0, 0, 0]
    return xenserver_version