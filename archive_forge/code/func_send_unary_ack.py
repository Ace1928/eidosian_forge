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
def send_unary_ack(self, ack_ids, ack_reqs_dict) -> Tuple[List[requests.AckRequest], List[requests.AckRequest]]:
    """Send a request using a separate unary request instead of over the stream.

        If a RetryError occurs, the manager shutdown is triggered, and the
        error is re-raised.
        """
    assert ack_ids
    assert len(ack_ids) == len(ack_reqs_dict)
    error_status = None
    ack_errors_dict = None
    try:
        self._client.acknowledge(subscription=self._subscription, ack_ids=ack_ids)
    except exceptions.GoogleAPICallError as exc:
        _LOGGER.debug('Exception while sending unary RPC. This is typically non-fatal as stream requests are best-effort.', exc_info=True)
        error_status = _get_status(exc)
        ack_errors_dict = _get_ack_errors(exc)
    except exceptions.RetryError as exc:
        exactly_once_delivery_enabled = self._exactly_once_delivery_enabled()
        for req in ack_reqs_dict.values():
            if req.future:
                if exactly_once_delivery_enabled:
                    e = AcknowledgeError(AcknowledgeStatus.OTHER, 'RetryError while sending ack RPC.')
                    req.future.set_exception(e)
                else:
                    req.future.set_result(AcknowledgeStatus.SUCCESS)
        _LOGGER.debug('RetryError while sending ack RPC. Waiting on a transient error resolution for too long, will now trigger shutdown.', exc_info=False)
        self._on_rpc_done(exc)
        raise
    if self._exactly_once_delivery_enabled():
        requests_completed, requests_to_retry = _process_requests(error_status, ack_reqs_dict, ack_errors_dict)
    else:
        requests_completed = []
        requests_to_retry = []
        for req in ack_reqs_dict.values():
            if req.future:
                req.future.set_result(AcknowledgeStatus.SUCCESS)
            requests_completed.append(req)
    return (requests_completed, requests_to_retry)