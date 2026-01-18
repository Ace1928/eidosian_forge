import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.common.google import GoogleResponse, GoogleBaseConnection, ResourceNotFoundError
def ex_bulk_record_changes(self, zone, records):
    """
        Bulk add and delete records.

        :param zone: Zone where the requested record changes are done.
        :type  zone: :class:`Zone`

        :param records: Dictionary of additions list or deletions list, or both
        of resourceRecordSets. For example:
            {'additions': [{'rrdatas': ['127.0.0.1'],
                            'kind': 'dns#resourceRecordSet',
                            'type': 'A',
                            'name': 'www.example.com.',
                            'ttl': '300'}],
             'deletions': [{'rrdatas': ['127.0.0.1'],
                            'kind': 'dns#resourceRecordSet',
                            'type': 'A',
                            'name': 'www2.example.com.',
                            'ttl': '300'}]}
        :type  records: ``dict``

        :return: A dictionary of Record additions and deletions.
        :rtype: ``dict`` of additions and deletions of :class:`Record`
        """
    request = '/managedZones/%s/changes' % zone.id
    response = self.connection.request(request, method='POST', data=records).object
    response = response or {}
    response_data = {'additions': self._to_records(response.get('additions', []), zone), 'deletions': self._to_records(response.get('deletions', []), zone)}
    return response_data