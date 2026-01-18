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
def pre_connect_hook(self, params, headers):
    time_string = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    headers['Date'] = time_string
    tmp = []
    signature = self._get_aws_auth_b64(self.key, time_string)
    auth = {'AWSAccessKeyId': self.user_id, 'Signature': signature, 'Algorithm': 'HmacSHA1'}
    for k, v in auth.items():
        tmp.append('{}={}'.format(k, v))
    headers['X-Amzn-Authorization'] = 'AWS3-HTTPS ' + ','.join(tmp)
    return (params, headers)