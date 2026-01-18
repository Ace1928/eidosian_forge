from __future__ import (absolute_import, division, print_function)
import abc
from ansible.module_utils import six
from ansible_collections.community.dns.plugins.module_utils.zone import (
def delete_records(self, records_per_zone_id, stop_early_on_errors=True):
    """
        Delete multiple records.

        @param records_per_zone_id: Maps a zone ID to a list of DNS records (DNSRecord)
        @param stop_early_on_errors: If set to ``True``, try to stop changes after the first error happens.
                                     This might only work on some APIs.
        @return A dictionary mapping zone IDs to lists of tuples ``(record, deleted, failed)``.
                In case ``record`` was deleted or not deleted, ``deleted`` is ``True``
                respectively ``False``, and ``failed`` is ``None``. In case an error happened
                while deleting, ``deleted`` is ``False`` and ``failed`` is a ``DNSAPIError``
                instance hopefully providing information on the error.
        """
    results_per_zone_id = {}
    for zone_id, records in records_per_zone_id.items():
        result = []
        results_per_zone_id[zone_id] = result
        for record in records:
            try:
                result.append((record, self.delete_record(zone_id, record), None))
            except DNSAPIError as e:
                result.append((record, False, e))
                if stop_early_on_errors:
                    return results_per_zone_id
    return results_per_zone_id