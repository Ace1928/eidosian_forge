from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import fetch_url, open_url
import json
import time
def plugin_open_url_json(plugin, url, method='GET', timeout=10, data=None, headers=None, accept_errors=None, allow_empty_result=False, allowed_empty_result_status_codes=(200, 204), templar=None):
    """
    Make general request to Hetzner's JSON robot API.
    """
    user = plugin.get_option('hetzner_user')
    password = plugin.get_option('hetzner_password')
    if templar is not None:
        if templar.is_template(user):
            user = templar.template(variable=user, disable_lookups=False)
        if templar.is_template(password):
            password = templar.template(variable=password, disable_lookups=False)
    try:
        response = open_url(url, url_username=user, url_password=password, force_basic_auth=True, data=data, headers=headers, method=method, timeout=timeout)
        status = response.code
        content = response.read()
    except HTTPError as e:
        status = e.code
        try:
            content = e.read()
        except AttributeError:
            content = b''
    except Exception as e:
        raise PluginException('Failed request to Hetzner Robot server endpoint {0}: {1}'.format(url, e))
    if not content:
        if allow_empty_result and status in allowed_empty_result_status_codes:
            return (None, None)
        raise PluginException('Cannot retrieve content from {0}, HTTP status code {1}'.format(url, status))
    try:
        result = json.loads(content.decode('utf-8'))
        if 'error' in result:
            if accept_errors:
                if result['error']['code'] in accept_errors:
                    return (result, result['error']['code'])
            raise PluginException(format_error_msg(result['error']))
        return (result, None)
    except ValueError:
        raise PluginException('Cannot decode content retrieved from {0}'.format(url))