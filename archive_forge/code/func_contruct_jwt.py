import sys
import json
import time
import base64
import unittest
from unittest import mock
import libcloud.common.gig_g8
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, StorageVolume
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gig_g8 import G8Network, G8NodeDriver, G8PortForward
def contruct_jwt(data):
    jsondata = json.dumps(data).encode()
    return 'header.{}.signature'.format(base64.encodebytes(jsondata).decode())