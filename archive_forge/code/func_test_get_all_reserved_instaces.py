from tests.unit import AWSMockServiceTestCase
from boto.ec2.connection import EC2Connection
from boto.ec2.reservedinstance import ReservedInstance
def test_get_all_reserved_instaces(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_all_reserved_instances()
    self.assertEqual(len(response), 1)
    self.assertTrue(isinstance(response[0], ReservedInstance))
    self.assertEquals(response[0].id, 'ididididid')
    self.assertEquals(response[0].instance_count, 5)
    self.assertEquals(response[0].start, '2014-05-03T14:10:10.944Z')
    self.assertEquals(response[0].end, '2014-05-03T14:10:11.000Z')