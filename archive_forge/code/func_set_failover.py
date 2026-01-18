from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def set_failover(module, ip, value, timeout=180):
    """
    Set current routing target of failover IP.

    Return a pair ``(value, changed)``. The value ``None`` for ``value`` represents unrouted.

    See https://robot.your-server.de/doc/webservice/en.html#post-failover-failover-ip
    and https://robot.your-server.de/doc/webservice/en.html#delete-failover-failover-ip
    """
    url = '{0}/failover/{1}'.format(BASE_URL, ip)
    if value is None:
        result, error = fetch_url_json(module, url, method='DELETE', timeout=timeout, accept_errors=['FAILOVER_ALREADY_ROUTED'])
    else:
        headers = {'Content-type': 'application/x-www-form-urlencoded'}
        data = dict(active_server_ip=value)
        result, error = fetch_url_json(module, url, method='POST', timeout=timeout, data=urlencode(data), headers=headers, accept_errors=['FAILOVER_ALREADY_ROUTED'])
    if error is not None:
        return (value, False)
    else:
        return (result['failover']['active_server_ip'], True)