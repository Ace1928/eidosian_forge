import warnings
from billiard.common import TERM_SIGNAME
from kombu.matcher import match
from kombu.pidbox import Mailbox
from kombu.utils.compat import register_after_fork
from kombu.utils.functional import lazy
from kombu.utils.objects import cached_property
from celery.exceptions import DuplicateNodenameWarning
from celery.utils.log import get_logger
from celery.utils.text import pluralize
def scheduled(self, safe=None):
    """Return list of scheduled tasks with details.

        Returns:
            Dict: Dictionary ``{HOSTNAME: [TASK_SCHEDULED_INFO,...]}``.

        Here is the list of ``TASK_SCHEDULED_INFO`` fields:

        * ``eta`` - scheduled time for task execution as string in ISO 8601 format
        * ``priority`` - priority of the task
        * ``request`` - field containing ``TASK_INFO`` value.

        See Also:
            For more details about ``TASK_INFO``  see :func:`query_task` return value.
        """
    return self._request('scheduled')