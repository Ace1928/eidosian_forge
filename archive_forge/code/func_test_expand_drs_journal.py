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
def test_expand_drs_journal(driver):
    cgs = driver.ex_list_consistency_groups(name='sdk_test2_cg')
    cg_id = cgs[0].id
    expand_by = '100'
    result = driver.ex_expand_journal(cg_id, expand_by)
    assert result is True