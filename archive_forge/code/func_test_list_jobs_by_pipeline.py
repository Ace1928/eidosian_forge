import time
from boto.elastictranscoder.layer1 import ElasticTranscoderConnection
from boto.elastictranscoder.exceptions import ValidationException
from tests.compat import unittest
import boto.s3
import boto.sns
import boto.iam
import boto.sns
def test_list_jobs_by_pipeline(self):
    pipeline_id = self.create_pipeline()
    response = self.api.list_jobs_by_pipeline(pipeline_id)
    self.assertEqual(response['Jobs'], [])