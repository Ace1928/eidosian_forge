from typing import AsyncIterator, Mapping, Optional
from google.cloud.pubsub_v1.types import BatchSettings
from google.cloud.pubsublite.admin_client import AdminClient
from google.cloud.pubsublite.internal.endpoints import regional_endpoint
from google.cloud.pubsublite.internal.publisher_client_id import PublisherClientId
from google.cloud.pubsublite.internal.publish_sequence_number import (
from google.cloud.pubsublite.internal.wire.client_cache import ClientCache
from google.cloud.pubsublite.internal.wire.default_routing_policy import (
from google.cloud.pubsublite.internal.wire.gapic_connection import (
from google.cloud.pubsublite.internal.wire.merge_metadata import merge_metadata
from google.cloud.pubsublite.internal.wire.partition_count_watcher_impl import (
from google.cloud.pubsublite.internal.wire.partition_count_watching_publisher import (
from google.cloud.pubsublite.internal.wire.publisher import Publisher
from google.cloud.pubsublite.internal.wire.single_partition_publisher import (
from google.cloud.pubsublite.types import Partition, TopicPath
from google.cloud.pubsublite.internal.routing_metadata import topic_routing_metadata
from google.cloud.pubsublite_v1 import InitialPublishRequest, PublishRequest
from google.cloud.pubsublite_v1.services.publisher_service import async_client
from google.api_core.client_options import ClientOptions
from google.auth.credentials import Credentials
def policy_factory(partition_count: int):
    return DefaultRoutingPolicy(partition_count)