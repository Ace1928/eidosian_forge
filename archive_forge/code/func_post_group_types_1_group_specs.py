from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def post_group_types_1_group_specs(self, body, **kw):
    assert list(body) == ['group_specs']
    return (200, {}, {'group_specs': {'k': 'v'}})