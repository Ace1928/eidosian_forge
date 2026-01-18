import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_logbook_add_task_detail(self):
    lb_id = uuidutils.generate_uuid()
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    fd = models.FlowDetail('test', uuid=uuidutils.generate_uuid())
    td = models.TaskDetail('detail-1', uuid=uuidutils.generate_uuid())
    td.version = '4.2'
    fd.add(td)
    lb.add(fd)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
    with contextlib.closing(self._get_connection()) as conn:
        lb2 = conn.get_logbook(lb_id)
        self.assertEqual(1, len(lb2))
        tasks = 0
        for fd in lb:
            tasks += len(fd)
        self.assertEqual(1, tasks)
    with contextlib.closing(self._get_connection()) as conn:
        lb2 = conn.get_logbook(lb_id)
        fd2 = lb2.find(fd.uuid)
        td2 = fd2.find(td.uuid)
        self.assertIsNotNone(td2)
        self.assertEqual('detail-1', td2.name)
        self.assertEqual('4.2', td2.version)
        self.assertEqual(states.EXECUTE, td2.intention)