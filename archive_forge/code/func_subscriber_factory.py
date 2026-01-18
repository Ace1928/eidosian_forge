from typing import Optional, Mapping, Set, AsyncIterator, Callable
from uuid import uuid4
from google.api_core.client_options import ClientOptions
from google.auth.credentials import Credentials
from google.cloud.pubsublite.cloudpubsub.reassignment_handler import (
from google.cloud.pubsublite.cloudpubsub.message_transforms import (
from google.cloud.pubsublite.internal.wire.client_cache import ClientCache
from google.cloud.pubsublite.types import FlowControlSettings
from google.cloud.pubsublite.cloudpubsub.internal.ack_set_tracker_impl import (
from google.cloud.pubsublite.cloudpubsub.internal.assigning_subscriber import (
from google.cloud.pubsublite.cloudpubsub.internal.single_partition_subscriber import (
from google.cloud.pubsublite.cloudpubsub.message_transformer import MessageTransformer
from google.cloud.pubsublite.cloudpubsub.nack_handler import (
from google.cloud.pubsublite.cloudpubsub.internal.single_subscriber import (
from google.cloud.pubsublite.internal.endpoints import regional_endpoint
from google.cloud.pubsublite.internal.wire.assigner import Assigner
from google.cloud.pubsublite.internal.wire.assigner_impl import AssignerImpl
from google.cloud.pubsublite.internal.wire.committer_impl import CommitterImpl
from google.cloud.pubsublite.internal.wire.fixed_set_assigner import FixedSetAssigner
from google.cloud.pubsublite.internal.wire.gapic_connection import (
from google.cloud.pubsublite.internal.wire.merge_metadata import merge_metadata
from google.cloud.pubsublite.internal.wire.pubsub_context import pubsub_context
import google.cloud.pubsublite.internal.wire.subscriber_impl as wire_subscriber
from google.cloud.pubsublite.internal.wire.subscriber_reset_handler import (
from google.cloud.pubsublite.types import Partition, SubscriptionPath
from google.cloud.pubsublite.internal.routing_metadata import (
from google.cloud.pubsublite_v1 import (
from google.cloud.pubsublite_v1.services.subscriber_service.async_client import (
from google.cloud.pubsublite_v1.services.partition_assignment_service.async_client import (
from google.cloud.pubsublite_v1.services.cursor_service.async_client import (
def subscriber_factory(reset_handler: SubscriberResetHandler):
    return wire_subscriber.SubscriberImpl(InitialSubscribeRequest(subscription=str(subscription), partition=partition.value), _DEFAULT_FLUSH_SECONDS, GapicConnectionFactory(subscribe_connection_factory), reset_handler)