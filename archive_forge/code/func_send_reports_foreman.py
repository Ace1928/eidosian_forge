from __future__ import (absolute_import, division, print_function)
import os
from datetime import datetime
from collections import defaultdict
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.parsing.convert_bool import boolean as to_bool
from ansible.plugins.callback import CallbackBase
def send_reports_foreman(self, stats):
    """
        Send reports to Foreman to be parsed by its config report
        importer. The data is in a format that Foreman can handle
        without writing another report importer.
        """
    for host in stats.processed.keys():
        total = stats.summarize(host)
        report = {'config_report': {'host': host, 'reported_at': get_now(), 'metrics': {'time': {'total': int(get_time() - self.start_time)}}, 'status': {'applied': total['changed'], 'failed': total['failures'] + total['unreachable'], 'skipped': total['skipped']}, 'logs': list(build_log_foreman(self.items[host])), 'reporter': 'ansible', 'check_mode': self.check_mode}}
        if self.check_mode:
            report['config_report']['status']['pending'] = total['changed']
            report['config_report']['status']['applied'] = 0
        self._send_data('report', 'foreman', host, report)
        self.items[host] = []