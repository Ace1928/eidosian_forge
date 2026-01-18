import time
import kombu
from kombu.common import maybe_declare
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from celery import states
from celery._state import current_task, task_join_will_block
from . import base
from .asynchronous import AsyncBackendMixin, BaseResultConsumer
def destination_for(self, task_id, request):
    """Get the destination for result by task id.

        Returns:
            Tuple[str, str]: tuple of ``(reply_to, correlation_id)``.
        """
    try:
        request = request or current_task.request
    except AttributeError:
        raise RuntimeError(f'RPC backend missing task request for {task_id!r}')
    return (request.reply_to, request.correlation_id or task_id)