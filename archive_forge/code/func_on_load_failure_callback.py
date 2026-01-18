import stevedore
from testtools import matchers
from glance_store import backend
from glance_store.tests import base
def on_load_failure_callback(*args, **kwargs):
    raise