import contextlib
import email
from unittest import mock
import uuid
from heat.common import exception as exc
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_get_message_multi_part_bad_format(self):
    parts = [{'config': '1e0e5a60-2843-4cfd-9137-d90bdf18eef5', 'type': 'multipart'}, {'config': '9cab10ef-16ce-4be9-8b25-a67b7313eddb', 'type': 'text'}]
    self.init_config(parts=parts)
    self.rpc_client.show_software_config.return_value = {'config': '#!/bin/bash'}
    result = self.config.get_message()
    self.assertEqual('1e0e5a60-2843-4cfd-9137-d90bdf18eef5', self.rpc_client.show_software_config.call_args_list[0][0][1])
    self.assertEqual('9cab10ef-16ce-4be9-8b25-a67b7313eddb', self.rpc_client.show_software_config.call_args_list[1][0][1])
    message = email.message_from_string(result)
    self.assertTrue(message.is_multipart())
    subs = message.get_payload()
    self.assertEqual(1, len(subs))
    self.assertEqual('#!/bin/bash', subs[0].get_payload())