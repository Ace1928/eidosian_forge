from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_group_types_1(self, **kw):
    return (200, {}, {'group_type': {'id': 1, 'name': 'test-type-1', 'description': 'test_type-1-desc', 'group_specs': {'key': 'value'}}})