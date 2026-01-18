from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
def svc_run_command(self, cmd, cmdopts, cmdargs, timeout=10):
    """ Generic execute a SVC command
        :param cmd: svc command to run
        :type cmd: string
        :param cmdopts: svc command options, name parameter and value
        :type cmdopts: dict
        :param cmdargs: svc command arguments, non-named parameters
        :type cmdargs: list
        :param timeout: open_url argument to set timeout for http gateway
        :type timeout: int
        :returns: command output
        """
    rest = self._svc_token_wrap(cmd, cmdopts, cmdargs, timeout)
    self.log('svc_run_command rest=%s', rest)
    if rest['err']:
        msg = rest
        self.module.fail_json(msg=msg)
    return rest['out']