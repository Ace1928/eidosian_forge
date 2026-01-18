from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_types_1_extra_specs(self, body, **kw):
    assert list(body) == ['extra_specs']
    return (200, {}, {'extra_specs': {'k': 'v'}})