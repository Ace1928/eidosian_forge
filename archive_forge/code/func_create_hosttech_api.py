from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.dns.plugins.module_utils.argspec import (
from ansible_collections.community.dns.plugins.module_utils.provider import (
from ansible_collections.community.dns.plugins.module_utils.wsdl import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
from ansible_collections.community.dns.plugins.module_utils.hosttech.wsdl_api import (
from ansible_collections.community.dns.plugins.module_utils.hosttech.json_api import (
def create_hosttech_api(option_provider, http_helper):
    username = option_provider.get_option('hosttech_username')
    password = option_provider.get_option('hosttech_password')
    if username is not None and password is not None:
        if not HAS_LXML_ETREE:
            raise DNSAPIError('Needs lxml Python module (pip install lxml)')
        return HostTechWSDLAPI(http_helper, username, password, debug=False)
    token = option_provider.get_option('hosttech_token')
    if token is not None:
        return HostTechJSONAPI(http_helper, token)
    raise DNSAPIError('One of hosttech_token or both hosttech_username and hosttech_password must be provided!')