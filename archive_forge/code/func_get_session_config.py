from __future__ import absolute_import, division, print_function
import json
import os
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def get_session_config(self, config, commit=False, replace=False):
    """Loads the configuration onto the remote devices

        If the device doesn't support configuration sessions, this will
        fallback to using configure() to load the commands.  If that happens,
        there will be no returned diff or session values
        """
    resp = ''
    use_session = os.getenv('ANSIBLE_EOS_USE_SESSIONS', True)
    try:
        use_session = int(use_session)
    except ValueError:
        pass
    if not all((bool(use_session), self.supports_sessions)):
        if commit:
            return self.configure(config)
        else:
            self._module.warn('EOS can not check config without config session')
            result = {'changed': True}
            return result
    session = session_name()
    result = {'session': session}
    commands = ['configure session %s' % session]
    if replace:
        commands.append('rollback clean-config')
    commands.extend(config)
    response = self._connection.send_request(commands)
    if 'error' in response:
        commands = ['configure session %s' % session, 'abort']
        self._connection.send_request(commands)
        err = response['error']
        error_text = []
        for data in err['data']:
            error_text.extend(data.get('errors', []))
        error_text = '\n'.join(error_text) or err['message']
        self._module.fail_json(msg=error_text, code=err['code'])
    commands = ['configure session %s' % session, 'show session-config']
    if commit:
        commands.append('commit')
    else:
        commands.append('abort')
    response = self._connection.send_request(commands, output='text')
    for out in response:
        if out:
            resp += out + ''
    return resp.rstrip()