from designateclient import exceptions
from designateclient.tests import base
def test_get_error_message_with_errors(self):
    expected_msg = "u'nodot.com' is not a 'domainname'"
    errors = {'errors': [{'path': ['name'], 'message': expected_msg, 'validator': 'format', 'validator_value': 'domainname'}]}
    self.response_dict['message'] = None
    self.response_dict['errors'] = errors
    remote_err = exceptions.RemoteError(**self.response_dict)
    self.assertEqual(expected_msg, remote_err.message)