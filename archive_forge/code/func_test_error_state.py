import argparse
import base64
import builtins
import collections
import datetime
import io
import os
from unittest import mock
import fixtures
from oslo_utils import timeutils
import testtools
import novaclient
from novaclient import api_versions
from novaclient import base
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
import novaclient.v2.shell
@mock.patch('novaclient.v2.shell.time')
def test_error_state(self, mock_time):
    fault_msg = 'Oops'
    updated_objects = (base.Resource(None, info={'status': 'error', 'fault': {'message': fault_msg}}), base.Resource(None, info={'status': 'error'}))
    poll_fn = mock.MagicMock(side_effect=updated_objects)
    action = 'some'
    self.assertRaises(exceptions.ResourceInErrorState, novaclient.v2.shell._poll_for_status, poll_fn=poll_fn, obj_id='uuuuuuuuuuuiiiiiiiii', final_ok_states=['ok'], poll_period='3', action=action, show_progress=True, silent=False)
    self.assertRaises(exceptions.ResourceInErrorState, novaclient.v2.shell._poll_for_status, poll_fn=poll_fn, obj_id='uuuuuuuuuuuiiiiiiiii', final_ok_states=['ok'], poll_period='3', action=action, show_progress=True, silent=False)