import asyncio
import json
import logging
import time
import grpc
from itertools import chain
import aiohttp.web
import ray._private.utils
from ray.dashboard.consts import GCS_RPC_TIMEOUT_SECONDS
from ray.autoscaler._private.util import (
import ray.dashboard.consts as dashboard_consts
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray._private import ray_constants
from ray.core.generated import (
from ray.dashboard.datacenter import DataOrganizer, DataSource
from ray.dashboard.modules.node import node_consts
from ray.dashboard.modules.node.node_consts import (
from ray._private.ray_constants import (
from ray.dashboard.utils import async_loop_forever
def node_stats_to_dict(message):
    decode_keys = {'actorId', 'jobId', 'taskId', 'parentTaskId', 'sourceActorId', 'callerId', 'rayletId', 'workerId', 'placementGroupId'}
    core_workers_stats = message.core_workers_stats
    message.ClearField('core_workers_stats')
    try:
        result = dashboard_utils.message_to_dict(message, decode_keys)
        result['coreWorkersStats'] = [dashboard_utils.message_to_dict(m, decode_keys, always_print_fields_with_no_presence=True) for m in core_workers_stats]
        return result
    finally:
        message.core_workers_stats.extend(core_workers_stats)