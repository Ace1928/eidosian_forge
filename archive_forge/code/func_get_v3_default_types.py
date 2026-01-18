from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_v3_default_types(self, **kw):
    default_types = stub_default_types()
    return (200, {}, default_types)