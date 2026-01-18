import os
import flask
from . import exceptions
from ._utils import AttributeDict
def pathname_configs(url_base_pathname=None, routes_pathname_prefix=None, requests_pathname_prefix=None):
    _pathname_config_error_message = '\n    {} This is ambiguous.\n    To fix this, set `routes_pathname_prefix` instead of `url_base_pathname`.\n\n    Note that `requests_pathname_prefix` is the prefix for the AJAX calls that\n    originate from the client (the web browser) and `routes_pathname_prefix` is\n    the prefix for the API routes on the backend (this flask server).\n    `url_base_pathname` will set `requests_pathname_prefix` and\n    `routes_pathname_prefix` to the same value.\n    If you need these to be different values then you should set\n    `requests_pathname_prefix` and `routes_pathname_prefix`,\n    not `url_base_pathname`.\n    '
    url_base_pathname = get_combined_config('url_base_pathname', url_base_pathname)
    routes_pathname_prefix = get_combined_config('routes_pathname_prefix', routes_pathname_prefix)
    requests_pathname_prefix = get_combined_config('requests_pathname_prefix', requests_pathname_prefix)
    if url_base_pathname is not None and requests_pathname_prefix is not None:
        raise exceptions.InvalidConfig(_pathname_config_error_message.format('You supplied `url_base_pathname` and `requests_pathname_prefix`.'))
    if url_base_pathname is not None and routes_pathname_prefix is not None:
        raise exceptions.InvalidConfig(_pathname_config_error_message.format('You supplied `url_base_pathname` and `routes_pathname_prefix`.'))
    if url_base_pathname is not None and routes_pathname_prefix is None:
        routes_pathname_prefix = url_base_pathname
    elif routes_pathname_prefix is None:
        routes_pathname_prefix = '/'
    if not routes_pathname_prefix.startswith('/'):
        raise exceptions.InvalidConfig('`routes_pathname_prefix` needs to start with `/`')
    if not routes_pathname_prefix.endswith('/'):
        raise exceptions.InvalidConfig('`routes_pathname_prefix` needs to end with `/`')
    app_name = load_dash_env_vars().DASH_APP_NAME
    if not requests_pathname_prefix and app_name:
        requests_pathname_prefix = '/' + app_name + routes_pathname_prefix
    elif requests_pathname_prefix is None:
        requests_pathname_prefix = routes_pathname_prefix
    if not requests_pathname_prefix.startswith('/'):
        raise exceptions.InvalidConfig('`requests_pathname_prefix` needs to start with `/`')
    return (url_base_pathname, routes_pathname_prefix, requests_pathname_prefix)