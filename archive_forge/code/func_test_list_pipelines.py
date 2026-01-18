import time
from tests.unit import unittest
from boto.datapipeline import layer1
def test_list_pipelines(self):
    pipeline_id = self.create_pipeline('name5', 'unique_id5')
    pipeline_id_list = [p['id'] for p in self.connection.list_pipelines()['pipelineIdList']]
    self.assertTrue(pipeline_id in pipeline_id_list)