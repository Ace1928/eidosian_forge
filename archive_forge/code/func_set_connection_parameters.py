from __future__ import absolute_import, division, print_function
import json
import re
import traceback
from ansible.module_utils.six import PY3
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import ConnectionError
from ansible.plugins.httpapi import HttpApiBase
from copy import copy
def set_connection_parameters(self):
    connection_parameters = {}
    for key in CONNECTION_KEYS:
        if key == 'login_domain':
            value = self.params.get(key) if self.params.get(key) is not None else self.get_option(CONNECTION_MAP.get(key, key))
            self.set_option(key, value)
        else:
            value = self.params.get(key) if self.params.get(key) is not None else self.connection.get_option(CONNECTION_MAP.get(key, key))
            self.connection.set_option(CONNECTION_MAP.get(key, key), value)
        connection_parameters[key] = value
        if value != self.connection_parameters.get(key) and key in RESET_KEYS:
            self.connection._connected = False
            self.connection.queue_message('vvvv', "set_connection_parameters() - resetting connection due to '{0}' change".format(key))
    if self.connection_parameters != connection_parameters:
        self.connection_parameters = copy(connection_parameters)
        connection_parameters.pop('password')
        msg = 'set_connection_parameters() - changed connection parameters {0}'.format(connection_parameters)
        self.connection.queue_message('vvvv', msg)