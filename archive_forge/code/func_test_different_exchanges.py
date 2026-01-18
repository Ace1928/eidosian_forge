import os
import requests
import subprocess
import time
import uuid
import concurrent.futures
from oslo_config import cfg
from testtools import matchers
import oslo_messaging
from oslo_messaging.tests.functional import utils
def test_different_exchanges(self):
    group1 = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url, use_fanout_ctrl=True))
    group2 = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url, exchange='a', use_fanout_ctrl=True))
    group3 = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url, exchange='b', use_fanout_ctrl=True))
    client1 = group1.client(1)
    data1 = [c for c in 'abcdefghijklmn']
    for i in data1:
        client1.append(text=i)
    client2 = group2.client()
    data2 = [c for c in 'opqrstuvwxyz']
    for i in data2:
        client2.append(text=i)
    actual1 = [[c for c in s.endpoint.sval] for s in group1.servers]
    self.assertThat(actual1, utils.IsValidDistributionOf(data1))
    actual1 = [c for c in group1.servers[1].endpoint.sval]
    self.assertThat([actual1], utils.IsValidDistributionOf(data1))
    for s in group1.servers:
        expected = len(data1) if group1.servers.index(s) == 1 else 0
        self.assertEqual(expected, len(s.endpoint.sval))
        self.assertEqual(0, s.endpoint.ival)
    actual2 = [[c for c in s.endpoint.sval] for s in group2.servers]
    for s in group2.servers:
        self.assertThat(len(s.endpoint.sval), matchers.GreaterThan(0))
        self.assertEqual(0, s.endpoint.ival)
    self.assertThat(actual2, utils.IsValidDistributionOf(data2))
    for s in group3.servers:
        self.assertEqual(0, len(s.endpoint.sval))
        self.assertEqual(0, s.endpoint.ival)