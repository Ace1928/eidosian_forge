import asyncio
import logging
import os
import time
from collections import deque
import aiohttp.web
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray._private.gcs_pubsub import GcsAioActorSubscriber
from ray.core.generated import (
from ray.dashboard.datacenter import DataSource, DataOrganizer
from ray.dashboard.modules.actor import actor_consts
from ray.dashboard.optional_utils import rest_response
def get_internal_states(self):
    states = {'total_published_events': self.total_published_events, 'total_dead_actors': len(self.dead_actors_queue), 'total_actors': len(DataSource.actors), 'queue_size': self.subscriber_queue_size}
    if self.accumulative_event_processing_s > 0:
        states['event_processing_per_s'] = self.total_published_events / self.accumulative_event_processing_s
    return states