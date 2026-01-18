import boto.swf.layer2
from boto.swf.layer2 import Domain, ActivityType, WorkflowType, WorkflowExecution
from tests.unit import unittest
from mock import Mock
def test_domain_list_workflows(self):
    self.domain._swf.list_workflow_types.return_value = {'typeInfos': [{'creationDate': 1332853651.136, 'description': 'Image processing sample workflow type', 'status': 'REGISTERED', 'workflowType': {'name': 'ProcessFile', 'version': '1.0'}}, {'creationDate': 1333551719.89, 'status': 'REGISTERED', 'workflowType': {'name': 'test_workflow_name', 'version': 'v1'}}]}
    expected_names = ('ProcessFile', 'test_workflow_name')
    workflow_types = self.domain.workflows()
    self.assertEquals(2, len(workflow_types))
    for workflow_type in workflow_types:
        self.assertIsInstance(workflow_type, WorkflowType)
        self.assertTrue(workflow_type.name in expected_names)
        self.assertEquals(self.domain.aws_access_key_id, workflow_type.aws_access_key_id)
        self.assertEquals(self.domain.aws_secret_access_key, workflow_type.aws_secret_access_key)
        self.assertEquals(self.domain.name, workflow_type.domain)
        self.assertEquals(self.domain.region, workflow_type.region)