from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
def test_external_running_here(self):
    self.config(run_external_periodic_tasks=True)

    class Manager(periodic_task.PeriodicTasks):

        @periodic_task.periodic_task(spacing=200, external_process_ok=True)
        def bar(self):
            return 'bar'
    m = Manager(self.conf)
    self.assertThat(m._periodic_tasks, matchers.HasLength(1))