import copy
import logging
import random
import time
from time import monotonic as now
from oslo_service._i18n import _
from oslo_service import _options
from oslo_utils import reflection
def run_periodic_tasks(self, context, raise_on_error=False):
    """Tasks to be run at a periodic interval."""
    idle_for = DEFAULT_INTERVAL
    for task_name, task in self._periodic_tasks:
        if task._periodic_external_ok and (not self.conf.run_external_periodic_tasks):
            continue
        cls_name = reflection.get_class_name(self, fully_qualified=False)
        full_task_name = '.'.join([cls_name, task_name])
        spacing = self._periodic_spacing[task_name]
        last_run = self._periodic_last_run[task_name]
        idle_for = min(idle_for, spacing)
        if last_run is not None:
            delta = last_run + spacing - now()
            if delta > 0:
                idle_for = min(idle_for, delta)
                continue
        LOG.debug('Running periodic task %(full_task_name)s', {'full_task_name': full_task_name})
        self._periodic_last_run[task_name] = _nearest_boundary(last_run, spacing)
        try:
            task(self, context)
        except BaseException:
            if raise_on_error:
                raise
            LOG.exception('Error during %(full_task_name)s', {'full_task_name': full_task_name})
        time.sleep(0)
    return idle_for