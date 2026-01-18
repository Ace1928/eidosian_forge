import base64
import logging
import os
import textwrap
import uuid
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import prettytable
from urllib import error
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient import exc
def event_log_formatter(events, event_log_context=None):
    """Return the events in log format."""
    event_log = []
    log_format = '%(event_time)s [%(rsrc_name)s]: %(rsrc_status)s  %(rsrc_status_reason)s'
    if event_log_context is None:
        event_log_context = EventLogContext()
    for event in events:
        rsrc_name = event_log_context.build_resource_name(event)
        event_time = getattr(event, 'event_time', '')
        log = log_format % {'event_time': event_time.replace('T', ' '), 'rsrc_name': rsrc_name, 'rsrc_status': getattr(event, 'resource_status', ''), 'rsrc_status_reason': getattr(event, 'resource_status_reason', '')}
        event_log.append(log)
    return '\n'.join(event_log)