import time
from tests.unit import unittest
from boto.datapipeline import layer1
def test_can_create_and_delete_a_pipeline(self):
    response = self.connection.create_pipeline('name', 'unique_id', 'description')
    self.connection.delete_pipeline(response['pipelineId'])