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
def query_task(self, *ids):
    """Return detail of tasks currently executed by workers.

        Arguments:
            *ids (str): IDs of tasks to be queried.

        Returns:
            Dict: Dictionary ``{HOSTNAME: {TASK_ID: [STATE, TASK_INFO]}}``.

        Here is the list of ``TASK_INFO`` fields:
            * ``id`` - ID of the task
            * ``name`` - Name of the task
            * ``args`` - Positinal arguments passed to the task
            * ``kwargs`` - Keyword arguments passed to the task
            * ``type`` - Type of the task
            * ``hostname`` - Hostname of the worker processing the task
            * ``time_start`` - Time of processing start
            * ``acknowledged`` - True when task was acknowledged to broker
            * ``delivery_info`` - Dictionary containing delivery information
                * ``exchange`` - Name of exchange where task was published
                * ``routing_key`` - Routing key used when task was published
                * ``priority`` - Priority used when task was published
                * ``redelivered`` - True if the task was redelivered
            * ``worker_pid`` - PID of worker processing the task

        """
    if len(ids) == 1 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return self._request('query_task', ids=ids)