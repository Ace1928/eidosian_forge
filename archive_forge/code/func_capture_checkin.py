import uuid
from sentry_sdk import Hub
from sentry_sdk._types import TYPE_CHECKING
def capture_checkin(monitor_slug=None, check_in_id=None, status=None, duration=None, monitor_config=None):
    check_in_event = _create_check_in_event(monitor_slug=monitor_slug, check_in_id=check_in_id, status=status, duration_s=duration, monitor_config=monitor_config)
    hub = Hub.current
    hub.capture_event(check_in_event)
    return check_in_event['check_in_id']