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
def test_get_consistency_group_by_id(driver):
    NttCisMockHttp.type = None
    cgs = driver.ex_list_consistency_groups()
    cg_id = [i for i in cgs if i.name == 'sdk_test2_cg'][0].id
    cg = driver.ex_get_consistency_group(cg_id)
    assert hasattr(cg, 'description')