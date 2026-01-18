from __future__ import absolute_import
import sys
import time
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk._compat import reraise
from sentry_sdk._functools import wraps
from sentry_sdk.crons import capture_checkin, MonitorStatus
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import BAGGAGE_HEADER_NAME, TRANSACTION_SOURCE_TASK
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def sentry_apply_entry(*args, **kwargs):
    scheduler, schedule_entry = args
    app = scheduler.app
    celery_schedule = schedule_entry.schedule
    monitor_name = schedule_entry.name
    hub = Hub.current
    integration = hub.get_integration(CeleryIntegration)
    if integration is None:
        return original_apply_entry(*args, **kwargs)
    if match_regex_list(monitor_name, integration.exclude_beat_tasks):
        return original_apply_entry(*args, **kwargs)
    with hub.configure_scope() as scope:
        scope.set_new_propagation_context()
        monitor_config = _get_monitor_config(celery_schedule, app, monitor_name)
        is_supported_schedule = bool(monitor_config)
        if is_supported_schedule:
            headers = schedule_entry.options.pop('headers', {})
            headers.update({'sentry-monitor-slug': monitor_name, 'sentry-monitor-config': monitor_config})
            check_in_id = capture_checkin(monitor_slug=monitor_name, monitor_config=monitor_config, status=MonitorStatus.IN_PROGRESS)
            headers.update({'sentry-monitor-check-in-id': check_in_id})
            schedule_entry.options['headers'] = headers
        return original_apply_entry(*args, **kwargs)