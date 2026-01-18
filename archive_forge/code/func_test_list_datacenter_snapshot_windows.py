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
def test_list_datacenter_snapshot_windows(driver):
    NttCisMockHttp.type = None
    ret = driver.list_snapshot_windows('f1d6a564-490e-4166-b91d-feddc1f92025', 'ADVANCED')
    assert isinstance(ret[0], dict)