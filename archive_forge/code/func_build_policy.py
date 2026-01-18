import base64
import datetime
import json
import weakref
import botocore
import botocore.auth
from botocore.awsrequest import create_request_object, prepare_request_dict
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import ArnParser, datetime2timestamp
from botocore.utils import fix_s3_host  # noqa
def build_policy(self, resource, date_less_than, date_greater_than=None, ip_address=None):
    """A helper to build policy.

        :type resource: str
        :param resource: The URL or the stream filename of the protected object

        :type date_less_than: datetime
        :param date_less_than: The URL will expire after the time has passed

        :type date_greater_than: datetime
        :param date_greater_than: The URL will not be valid until this time

        :type ip_address: str
        :param ip_address: Use 'x.x.x.x' for an IP, or 'x.x.x.x/x' for a subnet

        :rtype: str
        :return: The policy in a compact string.
        """
    moment = int(datetime2timestamp(date_less_than))
    condition = OrderedDict({'DateLessThan': {'AWS:EpochTime': moment}})
    if ip_address:
        if '/' not in ip_address:
            ip_address += '/32'
        condition['IpAddress'] = {'AWS:SourceIp': ip_address}
    if date_greater_than:
        moment = int(datetime2timestamp(date_greater_than))
        condition['DateGreaterThan'] = {'AWS:EpochTime': moment}
    ordered_payload = [('Resource', resource), ('Condition', condition)]
    custom_policy = {'Statement': [OrderedDict(ordered_payload)]}
    return json.dumps(custom_policy, separators=(',', ':'))