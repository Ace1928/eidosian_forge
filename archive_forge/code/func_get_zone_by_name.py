from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible_collections.community.dns.plugins.module_utils.argspec import (
from ansible_collections.community.dns.plugins.module_utils.json_api_helper import (
from ansible_collections.community.dns.plugins.module_utils.provider import (
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.zone import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
def get_zone_by_name(self, name):
    """
        Given a zone name, return the zone contents if found.

        @param name: The zone name (string)
        @return The zone information (DNSZone), or None if not found
        """
    result, info = self._get('v1/zones', expected=[200, 404], query=dict(name=name))
    for zone in result['zones']:
        if zone.get('name') == name:
            return _create_zone_from_json(zone)
    return None