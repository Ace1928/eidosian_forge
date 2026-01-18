from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.dns.plugins.module_utils.json_api_helper import (
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.zone import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
def get_zone_with_records_by_name(self, name, prefix=NOT_PROVIDED, record_type=NOT_PROVIDED):
    """
        Given a zone name, return the zone contents with records if found.

        @param name: The zone name (string)
        @param prefix: The prefix to filter for, if provided. Since None is a valid value,
                       the special constant NOT_PROVIDED indicates that we are not filtering.
        @param record_type: The record type to filter for, if provided
        @return The zone information with records (DNSZoneWithRecords), or None if not found
        """
    result = self._list_pagination('user/v1/zones', query=dict(query=name))
    for zone in result:
        if zone['name'] == name:
            result, info = self._get('user/v1/zones/{0}'.format(zone['id']), expected=[200])
            return _create_zone_with_records_from_json(result['data'], prefix=prefix, record_type=record_type)
    return None