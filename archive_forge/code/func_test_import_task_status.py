import datetime
from testtools import content as ttc
import time
from unittest import mock
import uuid
from oslo_log import log as logging
from oslo_utils import fixture as time_fixture
from oslo_utils import units
from glance.tests import functional
from glance.tests import utils as test_utils
@mock.patch('oslo_utils.timeutils.StopWatch.expired', new=lambda x: True)
def test_import_task_status(self):
    self.start_server()
    limit = 3 * units.Mi
    image_id = self._create_and_stage(data_iter=test_utils.FakeData(limit))
    statuses = []

    def grab_task_status():
        image = self.api_get('/v2/images/%s' % image_id).json
        task_id = image['os_glance_import_task']
        task = self.api_get('/v2/tasks/%s' % task_id).json
        msg = task['message']
        if msg not in statuses:
            statuses.append(msg)

    def fake_upload(data, *a, **k):
        while True:
            grab_task_status()
            if not data.read(65536):
                break
            time.sleep(0.1)
    with mock.patch('glance.location.ImageProxy._upload_to_store') as mu:
        mu.side_effect = fake_upload
        resp = self._import_direct(image_id, ['store2'])
        self.assertEqual(202, resp.status_code)
        for i in range(0, 100):
            image = self.api_get('/v2/images/%s' % image_id).json
            if not image.get('os_glance_import_task'):
                break
            time.sleep(0.1)
    self.assertEqual('active', image['status'])
    self.assertEqual(['', 'Copied 0 MiB', 'Copied 1 MiB', 'Copied 2 MiB', 'Copied 3 MiB'], statuses)