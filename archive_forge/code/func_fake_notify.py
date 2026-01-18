import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
def fake_notify(action, initiator, outcome, target, event_type, reason=None, **kwargs):
    service_security = cadftaxonomy.SERVICE_SECURITY
    event = eventfactory.EventFactory().new_event(eventType=cadftype.EVENTTYPE_ACTIVITY, outcome=outcome, action=action, initiator=initiator, target=target, reason=reason, observer=cadfresource.Resource(typeURI=service_security))
    for key, value in kwargs.items():
        setattr(event, key, value)
    note = {'action': action, 'initiator': initiator, 'event': event, 'event_type': event_type, 'send_notification_called': True}
    self._notifications.append(note)