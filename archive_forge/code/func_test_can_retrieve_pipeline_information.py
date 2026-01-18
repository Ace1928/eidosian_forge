import time
from boto.elastictranscoder.layer1 import ElasticTranscoderConnection
from boto.elastictranscoder.exceptions import ValidationException
from tests.compat import unittest
import boto.s3
import boto.sns
import boto.iam
import boto.sns
def test_can_retrieve_pipeline_information(self):
    pipeline_id = self.create_pipeline()
    pipelines = self.api.list_pipelines()['Pipelines']
    pipeline_names = [p['Name'] for p in pipelines]
    self.assertIn(self.pipeline_name, pipeline_names)
    response = self.api.read_pipeline(pipeline_id)
    self.assertEqual(response['Pipeline']['Id'], pipeline_id)