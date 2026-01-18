from collections import abc
import datetime
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import importutils
from glance.common import exception
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
def new_task_executor(self, context):
    try:
        task_executor = CONF.task.task_executor
        if task_executor == 'eventlet':
            if not TaskExecutorFactory.eventlet_deprecation_warned:
                msg = _LW('The `eventlet` executor has been deprecated. Use `taskflow` instead.')
                LOG.warning(msg)
                TaskExecutorFactory.eventlet_deprecation_warned = True
            task_executor = 'taskflow'
        executor_cls = 'glance.async_.%s_executor.TaskExecutor' % task_executor
        LOG.debug('Loading %s executor', task_executor)
        executor = importutils.import_class(executor_cls)
        return executor(context, self.task_repo, self.image_repo, self.image_factory, admin_repo=self.admin_repo)
    except ImportError:
        with excutils.save_and_reraise_exception():
            LOG.exception(_LE('Failed to load the %s executor provided in the config.'), CONF.task.task_executor)