import time
from boto.elastictranscoder.layer1 import ElasticTranscoderConnection
from boto.elastictranscoder.exceptions import ValidationException
from tests.compat import unittest
import boto.s3
import boto.sns
import boto.iam
import boto.sns
def test_create_delete_pipeline(self):
    pipeline = self.api.create_pipeline(self.pipeline_name, self.input_bucket, self.output_bucket, self.role_arn, {'Progressing': '', 'Completed': '', 'Warning': '', 'Error': ''})
    pipeline_id = pipeline['Pipeline']['Id']
    self.api.delete_pipeline(pipeline_id)