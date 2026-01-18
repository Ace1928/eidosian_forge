import uuid
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import service_token
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_invalidation(self):
    text = uuid.uuid4().hex
    test_url = 'http://test.example.com/abc'
    response_list = [{'status_code': 401}, {'text': text}]
    mock = self.requests_mock.get(test_url, response_list=response_list)
    resp = self.session.get(test_url)
    self.assertEqual(text, resp.text)
    self.assertEqual(2, mock.call_count)
    self.assertEqual(2, self.user_mock.call_count)
    self.assertEqual(2, self.service_mock.call_count)