import futurist
from oslo_log import log as logging
from glance.i18n import _LE
@staticmethod
def get_threadpool_executor_class():
    return futurist.ThreadPoolExecutor