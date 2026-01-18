import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_logbook_add_flow_detail(self):
    lb_id = uuidutils.generate_uuid()
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    fd = models.FlowDetail('test', uuid=uuidutils.generate_uuid())
    lb.add(fd)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
    with contextlib.closing(self._get_connection()) as conn:
        lb2 = conn.get_logbook(lb_id)
        self.assertEqual(1, len(lb2))
        self.assertEqual(1, len(lb))
        self.assertEqual(fd.name, lb2.find(fd.uuid).name)