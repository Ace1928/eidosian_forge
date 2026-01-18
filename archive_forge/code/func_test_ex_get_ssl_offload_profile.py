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
def test_ex_get_ssl_offload_profile(driver):
    profile_id = 'b1d3b5a7-75d7-4c44-a2b7-5bfa773dec63'
    profile = driver.ex_get_ssl_offload_profile(profile_id)
    assert profile.name == 'ssl_offload'