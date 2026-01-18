import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import tasks
def test_create_task(self):
    properties = {'type': 'import', 'input': {'import_from_format': 'ovf', 'import_from': 'swift://cloud.foo/myaccount/mycontainer/path'}}
    task = self.controller.create(**properties)
    self.assertEqual(_PENDING_ID, task.id)
    self.assertEqual('import', task.type)