import testtools
from unittest import mock
from glanceclient import exc
def test_handles_html(self):
    """exc.from_response should not print HTML."""
    mock_resp = mock.Mock()
    mock_resp.status_code = 404
    mock_resp.text = HTML_MSG
    mock_resp.headers = {'content-type': 'text/html'}
    err = exc.from_response(mock_resp, HTML_MSG)
    self.assertIsInstance(err, exc.HTTPNotFound)
    self.assertEqual('404 Entity Not Found: Entity could not be found', err.details)