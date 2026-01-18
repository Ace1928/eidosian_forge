from __future__ import (absolute_import, division, print_function)
import os
from datetime import datetime
from collections import defaultdict
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.parsing.convert_bool import boolean as to_bool
from ansible.plugins.callback import CallbackBase
def send_reports(self, stats):
    if self.report_type == 'foreman':
        self.send_reports_foreman(stats)
    elif self.report_type == 'proxy':
        self.send_reports_proxy_host_report(stats)
    else:
        self._display.warning(u'Unknown foreman endpoint type: {type}'.format(type=self.report_type))