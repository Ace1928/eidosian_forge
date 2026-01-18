from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
def svc_obj_info(self, cmd, cmdopts, cmdargs, timeout=10):
    """ Obtain information about an SVC object through the ls command
        :param cmd: svc command to run
        :type cmd: string
        :param cmdopts: svc command options, name parameter and value
        :type cmdopts: dict
        :param cmdargs: svc command arguments, non-named paramaters
        :type cmdargs: list
        :param timeout: open_url argument to set timeout for http gateway
        :type timeout: int
        :returns: command output
        :rtype: dict
        """
    rest = self._svc_token_wrap(cmd, cmdopts, cmdargs, timeout)
    self.log('svc_obj_info rest=%s', rest)
    if rest['code']:
        if rest['code'] == 500:
            return None
    if rest['err']:
        self.module.fail_json(msg=rest)
    return rest['out']