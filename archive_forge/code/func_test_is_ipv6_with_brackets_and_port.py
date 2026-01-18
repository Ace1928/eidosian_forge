from tests.compat import mock, unittest
import datetime
import hashlib
import hmac
import locale
import time
import boto.utils
from boto.utils import Password
from boto.utils import pythonize_name
from boto.utils import _build_instance_metadata_url
from boto.utils import get_instance_userdata
from boto.utils import retry_url
from boto.utils import LazyLoadMetadata
from boto.compat import json, _thread
def test_is_ipv6_with_brackets_and_port(self):
    hostname = '[bf1d:cb48:4513:d1f1:efdd:b290:9ff9:64be]:8080'
    result = boto.utils.host_is_ipv6(hostname)
    self.assertTrue(result)