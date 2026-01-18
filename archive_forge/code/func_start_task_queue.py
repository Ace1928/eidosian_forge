from __future__ import annotations
import abc
import time
import asyncio
import functools
from lazyops.imports._niquests import resolve_niquests
import niquests
from lazyops.libs.pooler import ThreadPooler
from lazyops.utils.logs import logger, null_logger, Logger
from lazyops.utils.times import Timer
from typing import Optional, Dict, Any, List, Union, Type, Set, Tuple, Callable, TypeVar, TYPE_CHECKING
from .config import PostHogSettings
from .utils import get_posthog_settings, register_posthog_client, get_posthog_client, has_existing_posthog_client
from .types import PostHogAuth, PostHogEndpoint, EventQueue, EventT
def start_task_queue(self, num_workers: Optional[int]=None):
    """
        Starts the Task Queue
        """
    if self.started:
        return
    if not self.enabled:
        self.logger.warning('PostHog is not enabled. Please set `POSTHOG_API_KEY` to enable PostHog')
        return
    num_workers = num_workers or self.settings.num_workers
    self.autologger.info(f'Starting PostHog Event Queue with |g|{num_workers}|e| Workers', colored=True)
    self.started_time = Timer()
    self.event_queue = EventQueue()
    self.main_task = asyncio.create_task(self.run_task_queue(num_workers=num_workers))