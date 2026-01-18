from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils._text import to_native
def iso8601_duration_from_seconds(seconds, format=None):
    check_for_import()
    try:
        duration = isodate.Duration(seconds=seconds)
        iso8601_duration = isodate.duration_isoformat(duration, format=isodate.D_DEFAULT if format is None else format)
    except Exception as exc:
        raise AnsibleFilterError('iso8601_duration_from_seconds - error: %s - received: %s' % (to_native(exc), seconds))
    return iso8601_duration