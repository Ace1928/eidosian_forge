from __future__ import division
import collections
import functools
import itertools
import logging
import threading
import typing
from typing import Any, Dict, Callable, Iterable, List, Optional, Set, Tuple
import uuid
import grpc  # type: ignore
from google.api_core import bidi
from google.api_core import exceptions
from google.cloud.pubsub_v1 import types
from google.cloud.pubsub_v1.subscriber._protocol import dispatcher
from google.cloud.pubsub_v1.subscriber._protocol import heartbeater
from google.cloud.pubsub_v1.subscriber._protocol import histogram
from google.cloud.pubsub_v1.subscriber._protocol import leaser
from google.cloud.pubsub_v1.subscriber._protocol import messages_on_hold
from google.cloud.pubsub_v1.subscriber._protocol import requests
from google.cloud.pubsub_v1.subscriber.exceptions import (
import google.cloud.pubsub_v1.subscriber.message
from google.cloud.pubsub_v1.subscriber import futures
from google.cloud.pubsub_v1.subscriber.scheduler import ThreadScheduler
from google.pubsub_v1 import types as gapic_types
from google.rpc.error_details_pb2 import ErrorInfo  # type: ignore
from google.rpc import code_pb2  # type: ignore
from google.rpc import status_pb2
def send_unary_modack(self, modify_deadline_ack_ids, modify_deadline_seconds, ack_reqs_dict, default_deadline=None) -> Tuple[List[requests.ModAckRequest], List[requests.ModAckRequest]]:
    """Send a request using a separate unary request instead of over the stream.

        If a RetryError occurs, the manager shutdown is triggered, and the
        error is re-raised.
        """
    assert modify_deadline_ack_ids
    assert modify_deadline_seconds is None or default_deadline is None
    error_status = None
    modack_errors_dict = None
    try:
        if default_deadline is None:
            deadline_to_ack_ids = collections.defaultdict(list)
            for n, ack_id in enumerate(modify_deadline_ack_ids):
                deadline = modify_deadline_seconds[n]
                deadline_to_ack_ids[deadline].append(ack_id)
            for deadline, ack_ids in deadline_to_ack_ids.items():
                self._client.modify_ack_deadline(subscription=self._subscription, ack_ids=ack_ids, ack_deadline_seconds=deadline)
        else:
            self._client.modify_ack_deadline(subscription=self._subscription, ack_ids=modify_deadline_ack_ids, ack_deadline_seconds=default_deadline)
    except exceptions.GoogleAPICallError as exc:
        _LOGGER.debug('Exception while sending unary RPC. This is typically non-fatal as stream requests are best-effort.', exc_info=True)
        error_status = _get_status(exc)
        modack_errors_dict = _get_ack_errors(exc)
    except exceptions.RetryError as exc:
        exactly_once_delivery_enabled = self._exactly_once_delivery_enabled()
        for req in ack_reqs_dict.values():
            if req.future:
                if exactly_once_delivery_enabled:
                    e = AcknowledgeError(AcknowledgeStatus.OTHER, 'RetryError while sending modack RPC.')
                    req.future.set_exception(e)
                else:
                    req.future.set_result(AcknowledgeStatus.SUCCESS)
        _LOGGER.debug('RetryError while sending modack RPC. Waiting on a transient error resolution for too long, will now trigger shutdown.', exc_info=False)
        self._on_rpc_done(exc)
        raise
    if self._exactly_once_delivery_enabled():
        requests_completed, requests_to_retry = _process_requests(error_status, ack_reqs_dict, modack_errors_dict)
    else:
        requests_completed = []
        requests_to_retry = []
        for req in ack_reqs_dict.values():
            if req.future:
                req.future.set_result(AcknowledgeStatus.SUCCESS)
            requests_completed.append(req)
    return (requests_completed, requests_to_retry)