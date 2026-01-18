import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_logbook_save_retrieve_many(self):
    lb_ids = {}
    for i in range(0, 10):
        lb_id = uuidutils.generate_uuid()
        lb_name = 'lb-%s-%s' % (i, lb_id)
        lb = models.LogBook(name=lb_name, uuid=lb_id)
        lb_ids[lb_id] = True
        with contextlib.closing(self._get_connection()) as conn:
            self.assertRaises(exc.NotFound, conn.get_logbook, lb_id)
            conn.save_logbook(lb)
    with contextlib.closing(self._get_connection()) as conn:
        lbs = conn.get_logbooks()
        for lb in lbs:
            self.assertIn(lb.uuid, lb_ids)
            lb_ids.pop(lb.uuid)
        self.assertEqual(0, len(lb_ids))