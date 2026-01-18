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
def rate_limit(self, task_name, rate_limit, destination=None, **kwargs):
    """Tell workers to set a new rate limit for task by type.

        Arguments:
            task_name (str): Name of task to change rate limit for.
            rate_limit (int, str): The rate limit as tasks per second,
                or a rate limit string (`'100/m'`, etc.
                see :attr:`celery.app.task.Task.rate_limit` for
                more information).

        See Also:
            :meth:`broadcast` for supported keyword arguments.
        """
    return self.broadcast('rate_limit', destination=destination, arguments={'task_name': task_name, 'rate_limit': rate_limit}, **kwargs)