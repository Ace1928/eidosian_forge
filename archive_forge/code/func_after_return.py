import sys
from billiard.einfo import ExceptionInfo, ExceptionWithTraceback
from kombu import serialization
from kombu.exceptions import OperationalError
from kombu.utils.uuid import uuid
from celery import current_app, states
from celery._state import _task_stack
from celery.canvas import _chain, group, signature
from celery.exceptions import Ignore, ImproperlyConfigured, MaxRetriesExceededError, Reject, Retry
from celery.local import class_property
from celery.result import EagerResult, denied_join_result
from celery.utils import abstract
from celery.utils.functional import mattrgetter, maybe_list
from celery.utils.imports import instantiate
from celery.utils.nodenames import gethostname
from celery.utils.serialization import raise_with_context
from .annotations import resolve_all as resolve_all_annotations
from .registry import _unpickle_task_v2
from .utils import appstr
def after_return(self, status, retval, task_id, args, kwargs, einfo):
    """Handler called after the task returns.

        Arguments:
            status (str): Current task state.
            retval (Any): Task return value/exception.
            task_id (str): Unique id of the task.
            args (Tuple): Original arguments for the task.
            kwargs (Dict): Original keyword arguments for the task.
            einfo (~billiard.einfo.ExceptionInfo): Exception information.

        Returns:
            None: The return value of this handler is ignored.
        """