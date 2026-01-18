import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import tasks
def test_list_tasks(self):
    tasks = self.controller.list()
    self.assertEqual(_PENDING_ID, tasks[0].id)
    self.assertEqual('import', tasks[0].type)
    self.assertEqual('pending', tasks[0].status)
    self.assertEqual(_PROCESSING_ID, tasks[1].id)
    self.assertEqual('import', tasks[1].type)
    self.assertEqual('processing', tasks[1].status)