import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import tasks
def test_list_tasks_with_desc_sort_dir(self):
    tasks = self.controller.list(sort_key='id', sort_dir='desc')
    self.assertEqual(2, len(tasks))
    self.assertEqual(_PENDING_ID, tasks[1].id)