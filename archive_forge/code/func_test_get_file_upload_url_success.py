from tests.unit import AWSMockServiceTestCase
from boto.mturk.connection import MTurkConnection
def test_get_file_upload_url_success(self):
    self.set_http_response(status_code=200, body=GET_FILE_UPLOAD_URL)
    rset = self.service_connection.get_file_upload_url('aid', 'qid')
    self.assertEquals(len(rset), 1)
    self.assertEquals(rset[0].FileUploadURL, 'http://s3.amazonaws.com/myawsbucket/puppy.jpg')