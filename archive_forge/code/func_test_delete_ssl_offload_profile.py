import os
import sys
import pytest
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import NTTCIS_PARAMS
from libcloud.common.nttcis import NttCisPool, NttCisVIPNode, NttCisPoolMember, NttCisAPIException
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.nttcis import NttCisLBDriver
def test_delete_ssl_offload_profile(driver):
    profile_name = 'ssl_offload'
    NttCisMockHttp.type = 'LIST'
    profile = driver.ex_list_ssl_offload_profiles(name=profile_name)[0]
    NttCisMockHttp.type = None
    result = driver.ex_delete_ssl_offload_profile(profile.id)
    assert result is True