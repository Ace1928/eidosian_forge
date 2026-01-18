from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def post_groups(self, **kw):
    group = _stub_group(id='1234', group_type='my_group_type', volume_types=['type1', 'type2'])
    return (202, {}, {'group': group})