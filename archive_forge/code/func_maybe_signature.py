import itertools
import operator
import warnings
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import MutableSequence
from copy import deepcopy
from functools import partial as _partial
from functools import reduce
from operator import itemgetter
from types import GeneratorType
from kombu.utils.functional import fxrange, reprcall
from kombu.utils.objects import cached_property
from kombu.utils.uuid import uuid
from vine import barrier
from celery._state import current_app
from celery.exceptions import CPendingDeprecationWarning
from celery.result import GroupResult, allow_join_result
from celery.utils import abstract
from celery.utils.collections import ChainMap
from celery.utils.functional import _regen
from celery.utils.functional import chunks as _chunks
from celery.utils.functional import is_list, maybe_list, regen, seq_concat_item, seq_concat_seq
from celery.utils.objects import getitem_property
from celery.utils.text import remove_repeating_from_task, truncate
def maybe_signature(d, app=None, clone=False):
    """Ensure obj is a signature, or None.

    Arguments:
        d (Optional[Union[abstract.CallableSignature, Mapping]]):
            Signature or dict-serialized signature.
        app (celery.Celery):
            App to bind signature to.
        clone (bool):
            If d' is already a signature, the signature
           will be cloned when this flag is enabled.

    Returns:
        Optional[abstract.CallableSignature]
    """
    if d is not None:
        if isinstance(d, abstract.CallableSignature):
            if clone:
                d = d.clone()
        elif isinstance(d, dict):
            d = signature(d)
        if app is not None:
            d._app = app
    return d