from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
def list_record_types(self):
    """
        >>> driver = DummyDNSDriver('key', 'secret')
        >>> driver.list_record_types()
        ['A']

        @inherits: :class:`DNSDriver.list_record_types`
        """
    return [RecordType.A]