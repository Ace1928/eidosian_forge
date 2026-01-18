import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_logbook_merge_flow_detail(self):
    lb_id = uuidutils.generate_uuid()
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    fd = models.FlowDetail('test', uuid=uuidutils.generate_uuid())
    lb.add(fd)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
    lb2 = models.LogBook(name=lb_name, uuid=lb_id)
    fd2 = models.FlowDetail('test2', uuid=uuidutils.generate_uuid())
    lb2.add(fd2)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb2)
    with contextlib.closing(self._get_connection()) as conn:
        lb3 = conn.get_logbook(lb_id)
        self.assertEqual(2, len(lb3))