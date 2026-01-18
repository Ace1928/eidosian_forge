import asyncio
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Callable, List, Tuple, Optional
import aiohttp.web
from aiohttp.web import Response
from abc import ABC, abstractmethod
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray.dashboard.consts import (
from ray.dashboard.datacenter import DataSource
from ray.dashboard.modules.log.log_manager import LogsManager
from ray.dashboard.optional_utils import rest_response
from ray.dashboard.state_aggregator import StateAPIManager
from ray.dashboard.utils import Change
from ray.util.state.common import (
from ray.util.state.exception import DataSourceUnavailable
from ray.util.state.state_manager import StateDataSourceClient
from ray.util.state.util import convert_string_to_type
@staticmethod
def enforce_max_concurrent_calls(func):
    """Decorator to enforce max number of invocations of the decorated func

        NOTE: This should be used as the innermost decorator if there are multiple
        ones.

        E.g., when decorating functions already with @routes.get(...), this must be
        added below then the routes decorators:
            ```
            @routes.get('/')
            @RateLimitedModule.enforce_max_concurrent_calls
            async def fn(self):
                ...

            ```
        """

    async def async_wrapper(self, *args, **kwargs):
        if self.max_num_call_ >= 0 and self.num_call_ >= self.max_num_call_:
            if self.logger_:
                self.logger_.warning(f'Max concurrent requests reached={self.max_num_call_}')
            return await self.limit_handler_()
        self.num_call_ += 1
        try:
            ret = await func(self, *args, **kwargs)
        finally:
            self.num_call_ -= 1
        return ret
    return async_wrapper