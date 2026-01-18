import logging
from typing import List, Optional, Union
from google.api_core.exceptions import InvalidArgument
from google.api_core.operation import Operation
from cloudsdk.google.protobuf.field_mask_pb2 import FieldMask  # pytype: disable=pyi-error
from google.cloud.pubsublite.admin_client_interface import AdminClientInterface
from google.cloud.pubsublite.types import (
from google.cloud.pubsublite.types.paths import ReservationPath
from google.cloud.pubsublite_v1 import (
def seek_subscription(self, subscription_path: SubscriptionPath, target: Union[BacklogLocation, PublishTime, EventTime]) -> Operation:
    request = SeekSubscriptionRequest(name=str(subscription_path))
    if isinstance(target, PublishTime):
        request.time_target = TimeTarget(publish_time=target.value)
    elif isinstance(target, EventTime):
        request.time_target = TimeTarget(event_time=target.value)
    elif isinstance(target, BacklogLocation):
        if target == BacklogLocation.END:
            request.named_target = SeekSubscriptionRequest.NamedTarget.HEAD
        else:
            request.named_target = SeekSubscriptionRequest.NamedTarget.TAIL
    else:
        raise InvalidArgument('A valid seek target must be specified.')
    return self._underlying.seek_subscription(request=request)