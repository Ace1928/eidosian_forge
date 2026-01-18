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
def test_list_cg_by_name(driver):
    NttCisMockHttp.type = 'CG_BY_NAME'
    name = 'sdk_test2_cg'
    cg = driver.ex_list_consistency_groups(name=name)
    assert cg[0].id == '195a426b-4559-4c79-849e-f22cdf2bfb6e'