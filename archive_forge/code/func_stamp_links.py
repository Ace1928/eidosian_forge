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
def stamp_links(self, visitor, append_stamps=False, **headers):
    """Stamp this signature links (callbacks and errbacks).
        Using a visitor will pass on responsibility for the stamping
        to the visitor.

        Arguments:
            visitor (StampingVisitor): Visitor API object.
            append_stamps (bool):
                If True, duplicated stamps will be appended to a list.
                If False, duplicated stamps will be replaced by the last stamp.
            headers (Dict): Stamps that should be added to headers.
        """
    non_visitor_headers = headers.copy()
    self_headers = False
    headers = deepcopy(non_visitor_headers)
    for link in maybe_list(self.options.get('link')) or []:
        link = maybe_signature(link, app=self.app)
        visitor_headers = None
        if visitor is not None:
            visitor_headers = visitor.on_callback(link, **headers) or {}
        headers = self._stamp_headers(visitor_headers=visitor_headers, append_stamps=append_stamps, self_headers=self_headers, **headers)
        link.stamp(visitor, append_stamps, **headers)
    headers = deepcopy(non_visitor_headers)
    for link in maybe_list(self.options.get('link_error')) or []:
        link = maybe_signature(link, app=self.app)
        visitor_headers = None
        if visitor is not None:
            visitor_headers = visitor.on_errback(link, **headers) or {}
        headers = self._stamp_headers(visitor_headers=visitor_headers, append_stamps=append_stamps, self_headers=self_headers, **headers)
        link.stamp(visitor, append_stamps, **headers)