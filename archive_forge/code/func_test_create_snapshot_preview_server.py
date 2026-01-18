import sys
import unittest
from types import GeneratorType
import pytest
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, httplib
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeLocation, NodeAuthPassword
from libcloud.test.secrets import NTTCIS_PARAMS
from libcloud.common.nttcis import (
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.nttcis import NttCisNic
from libcloud.compute.drivers.nttcis import NttCisNodeDriver as NttCis
def test_create_snapshot_preview_server(driver):
    snapshot_id = 'dd9a9e7e-2de7-4543-adef-bb1fda7ac030'
    server_name = 'test_snapshot'
    start = 'true'
    nic_connected = 'true'
    result = driver.ex_create_snapshot_preview_server(snapshot_id, server_name, start, nic_connected)
    assert result is True