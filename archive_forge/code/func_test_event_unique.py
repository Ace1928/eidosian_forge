import time
from unittest import mock
import uuid
from pycadf import attachment
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import event
from pycadf import geolocation
from pycadf import host
from pycadf import identifier
from pycadf import measurement
from pycadf import metric
from pycadf import reason
from pycadf import reporterstep
from pycadf import resource
from pycadf import tag
from pycadf.tests import base
from pycadf import timestamp
def test_event_unique(self):
    ev = event.Event(eventType='activity', initiator=resource.Resource(typeURI='storage'), action='read', target=resource.Resource(typeURI='storage'), observer=resource.Resource(id='target'), outcome='success')
    time.sleep(1)
    ev2 = event.Event(eventType='activity', initiator=resource.Resource(typeURI='storage'), action='read', target=resource.Resource(typeURI='storage'), observer=resource.Resource(id='target'), outcome='success')
    self.assertNotEqual(ev.id, ev2.id)
    self.assertNotEqual(ev.eventTime, ev2.eventTime)