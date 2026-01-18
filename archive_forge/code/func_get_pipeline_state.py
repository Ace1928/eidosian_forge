import time
from tests.unit import unittest
from boto.datapipeline import layer1
def get_pipeline_state(self, pipeline_id):
    response = self.connection.describe_pipelines([pipeline_id])
    for attr in response['pipelineDescriptionList'][0]['fields']:
        if attr['key'] == '@pipelineState':
            return attr['stringValue']