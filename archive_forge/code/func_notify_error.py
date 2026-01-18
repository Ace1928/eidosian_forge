import uuid
import webob
from oslo_messaging.notify import middleware
from oslo_messaging.tests import utils
from unittest import mock
def notify_error(context, publisher_id, event_type, priority, payload):
    raise Exception('error')