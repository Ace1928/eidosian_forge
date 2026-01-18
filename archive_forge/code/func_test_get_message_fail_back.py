import contextlib
import email
from unittest import mock
import uuid
from heat.common import exception as exc
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_get_message_fail_back(self):
    parts = [{'config': '2e0e5a60-2843-4cfd-9137-d90bdf18eef5', 'type': 'text'}]
    self.init_config(parts=parts)

    @contextlib.contextmanager
    def exc_filter():
        try:
            yield
        except exc.NotFound:
            pass
    self.rpc_client.ignore_error_by_name.return_value = exc_filter()
    self.rpc_client.show_software_config.side_effect = exc.NotFound()
    result = self.config.get_message()
    self.assertEqual('2e0e5a60-2843-4cfd-9137-d90bdf18eef5', self.rpc_client.show_software_config.call_args[0][1])
    message = email.message_from_string(result)
    self.assertTrue(message.is_multipart())
    subs = message.get_payload()
    self.assertEqual(1, len(subs))
    self.assertEqual('2e0e5a60-2843-4cfd-9137-d90bdf18eef5', subs[0].get_payload())