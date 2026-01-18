from __future__ import (absolute_import, division, print_function)
import logging
import json
import socket
from uuid import getnode
from ansible.plugins.callback import CallbackBase
from ansible.parsing.ajson import AnsibleJSONEncoder
def sendLog(self, host, category, logdata):
    options = {'app': 'ansible', 'meta': {'playbook': self.playbook_name, 'host': host, 'category': category}}
    logdata['info'].pop('invocation', None)
    warnings = logdata['info'].pop('warnings', None)
    if warnings is not None:
        self.flush({'warn': warnings}, options)
    self.flush(logdata, options)