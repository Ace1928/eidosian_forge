from celery import bootsteps
from celery.utils.log import get_logger
from celery.worker import pidbox
from .tasks import Tasks
def include_if(self, c):
    return c.app.conf.worker_enable_remote_control and c.conninfo.supports_exchange_type('fanout')