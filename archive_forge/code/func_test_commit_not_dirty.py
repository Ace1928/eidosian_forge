import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_commit_not_dirty(self):
    self.sot._body = mock.Mock()
    self.sot._body.dirty = dict()
    self.sot._header = mock.Mock()
    self.sot._header.dirty = dict()
    self.sot.commit(self.session)
    self.session.put.assert_not_called()