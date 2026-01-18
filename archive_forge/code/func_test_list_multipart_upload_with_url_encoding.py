from mock import patch, Mock
import unittest
from boto.s3.bucket import ResultSet
from boto.s3.bucketlistresultset import multipart_upload_lister
from boto.s3.bucketlistresultset import versioned_bucket_lister
def test_list_multipart_upload_with_url_encoding(self):
    self._test_patched_lister_encoding('get_all_multipart_uploads', multipart_upload_lister)