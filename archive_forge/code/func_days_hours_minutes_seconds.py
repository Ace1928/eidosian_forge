from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible.plugins.callback import CallbackBase
def days_hours_minutes_seconds(self, runtime):
    minutes = runtime.seconds // 60 % 60
    r_seconds = runtime.seconds % 60
    return (runtime.days, runtime.seconds // 3600, minutes, r_seconds)