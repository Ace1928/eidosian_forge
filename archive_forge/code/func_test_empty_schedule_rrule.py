from __future__ import absolute_import, division, print_function
import pytest
from ansible.errors import AnsibleError
from awx.main.models import JobTemplate, Schedule
from awx.api.serializers import SchedulePreviewSerializer
@pytest.mark.parametrize('freq', ('none', 'minute', 'hour', 'day', 'week', 'month'))
def test_empty_schedule_rrule(collection_import, freq):
    LookupModule = collection_import('plugins.lookup.schedule_rrule').LookupModule()
    if freq == 'day':
        pfreq = 'DAILY'
    elif freq == 'none':
        pfreq = 'DAILY;COUNT=1'
    else:
        pfreq = freq.upper() + 'LY'
    assert LookupModule.get_rrule(freq, {}).endswith(' RRULE:FREQ={0};INTERVAL=1'.format(pfreq))