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
def cancel_consumer(self, queue, destination=None, **kwargs):
    """Tell all (or specific) workers to stop consuming from ``queue``.

        See Also:
            Supports the same arguments as :meth:`broadcast`.
        """
    return self.broadcast('cancel_consumer', destination=destination, arguments={'queue': queue}, **kwargs)