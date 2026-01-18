from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_group_types_2(self, **kw):
    return (200, {}, {'group_type': {'id': 2, 'name': 'test-type-2', 'description': 'test_type-2-desc', 'group_specs': {}}})