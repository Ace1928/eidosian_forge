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
def as_execution_options(self):
    limit_hard, limit_soft = self.timelimit or (None, None)
    execution_options = {'task_id': self.id, 'root_id': self.root_id, 'parent_id': self.parent_id, 'group_id': self.group, 'group_index': self.group_index, 'shadow': self.shadow, 'chord': self.chord, 'chain': self.chain, 'link': self.callbacks, 'link_error': self.errbacks, 'expires': self.expires, 'soft_time_limit': limit_soft, 'time_limit': limit_hard, 'headers': self.headers, 'retries': self.retries, 'reply_to': self.reply_to, 'replaced_task_nesting': self.replaced_task_nesting, 'origin': self.origin}
    if hasattr(self, 'stamps') and hasattr(self, 'stamped_headers'):
        if self.stamps is not None and self.stamped_headers is not None:
            execution_options['stamped_headers'] = self.stamped_headers
            for k, v in self.stamps.items():
                execution_options[k] = v
    return execution_options