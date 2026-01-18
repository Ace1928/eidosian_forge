from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.dns.plugins.module_utils.argspec import (
from ansible_collections.community.dns.plugins.module_utils.provider import (
from ansible_collections.community.dns.plugins.module_utils.wsdl import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
from ansible_collections.community.dns.plugins.module_utils.hosttech.wsdl_api import (
from ansible_collections.community.dns.plugins.module_utils.hosttech.json_api import (
def create_hosttech_provider_information():
    return HosttechProviderInformation()