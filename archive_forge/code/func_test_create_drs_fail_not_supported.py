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
def test_create_drs_fail_not_supported(driver):
    NttCisMockHttp.type = 'FAIL_NOT_SUPPORTED'
    src_id = '032f3967-00e4-4780-b4ef-8587460f9dd4'
    target_id = 'aee58575-38e2-495f-89d3-854e6a886411'
    with pytest.raises(NttCisAPIException) as excinfo:
        driver.ex_create_consistency_group('sdk_cg', '100', src_id, target_id, description='A test consistency group')
    exception_msg = excinfo.value.msg
    assert exception_msg == 'DRS is not supported between source Data Center NA9 and target Data Center NA12.'