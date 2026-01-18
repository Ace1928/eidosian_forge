from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
@periodic_task.periodic_task(spacing=10 + periodic_task.DEFAULT_INTERVAL)
def doit_with_tocks(self, context):
    self.called['tocks'] += 1