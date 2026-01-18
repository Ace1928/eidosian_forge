import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_flow_detail_update_not_existing(self):
    lb_id = uuidutils.generate_uuid()
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    fd = models.FlowDetail('test', uuid=uuidutils.generate_uuid())
    lb.add(fd)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
    fd2 = models.FlowDetail('test-2', uuid=uuidutils.generate_uuid())
    lb.add(fd2)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
    with contextlib.closing(self._get_connection()) as conn:
        lb2 = conn.get_logbook(lb.uuid)
    self.assertIsNotNone(lb2.find(fd.uuid))
    self.assertIsNotNone(lb2.find(fd2.uuid))