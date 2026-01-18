import copy
import hmac
import uuid
import base64
import datetime
from hashlib import sha1
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import ET, b, httplib, urlencode
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.aws import AWSGenericResponse, AWSTokenConnection
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import LibcloudError
def ex_delete_all_records(self, zone):
    """
        Remove all the records for the provided zone.

        :param zone: Zone to delete records for.
        :type  zone: :class:`Zone`
        """
    deletions = []
    for r in zone.list_records():
        if r.type in (RecordType.NS, RecordType.SOA):
            continue
        deletions.append(('DELETE', r.name, r.type, r.data, r.extra))
    if deletions:
        self._post_changeset(zone, deletions)