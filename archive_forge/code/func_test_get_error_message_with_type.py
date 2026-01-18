from designateclient import exceptions
from designateclient.tests import base
def test_get_error_message_with_type(self):
    expected_msg = 'invalid_object'
    self.response_dict['message'] = None
    self.response_dict['errors'] = None
    self.response_dict['type'] = expected_msg
    remote_err = exceptions.RemoteError(**self.response_dict)
    self.assertEqual(expected_msg, remote_err.message)