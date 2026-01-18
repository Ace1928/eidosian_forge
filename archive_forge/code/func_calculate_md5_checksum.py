import io
import operator
import tempfile
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import _log
from openstack import exceptions
from openstack.image.v2 import image
from openstack.tests.unit import base
from openstack import utils
def calculate_md5_checksum(data):
    checksum = utils.md5(usedforsecurity=False)
    for chunk in data:
        checksum.update(chunk)
    return checksum.hexdigest()