from tests.unit import AWSMockServiceTestCase
from boto.glacier.layer1 import Layer1
from boto.glacier.response import GlacierResponse
def test_204_body_isnt_passed_to_json(self):
    response = self.create_response(status_code=204, header=[('Content-Type', 'application/json')])
    result = GlacierResponse(response, response.getheaders())
    self.assertEquals(result.status, response.status)