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
def test_ex_create_ssl_offload_profile(driver):
    net_domain_id = '6aafcf08-cb0b-432c-9c64-7371265db086'
    name = 'ssl_offload'
    domain_cert = driver.ex_list_ssl_domain_certs(name='alice')[0]
    result = driver.ex_create_ssl_offload_profile(net_domain_id, name, domain_cert.id, ciphers='!ECDHE+AES-GCM:')
    assert result is True