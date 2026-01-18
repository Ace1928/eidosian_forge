import http.client
from oslo_serialization import jsonutils
import webtest
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def restful_request(self, method='GET', headers=None, body=None, content_type=None, response_content_type=None, **kwargs):
    """Serialize/deserialize json as request/response body.

        .. WARNING::

            * Existing Accept header will be overwritten.
            * Existing Content-Type header will be overwritten.

        """
    headers = {} if not headers else headers
    body = self._to_content_type(body, headers, content_type)
    response = self.request(method=method, headers=headers, body=body, **kwargs)
    response_content_type = response_content_type or content_type
    self._from_content_type(response, content_type=response_content_type)
    if method != 'HEAD' and response.status_code >= http.client.BAD_REQUEST:
        self.assertValidErrorResponse(response)
    return response