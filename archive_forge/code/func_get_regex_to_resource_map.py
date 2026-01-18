import importlib
import random
import re
import warnings
def get_regex_to_resource_map():
    """Returns a dictionary with all of Jupyter Server's
    request handler URL regex patterns mapped to
    their resource name.

    e.g.
    { "/api/contents/<regex_pattern>": "contents", ...}
    """
    from jupyter_server.serverapp import JUPYTER_SERVICE_HANDLERS
    modules = []
    for mod_name in JUPYTER_SERVICE_HANDLERS.values():
        if mod_name:
            modules.extend(mod_name)
    resource_map = {}
    for handler_module in modules:
        mod = importlib.import_module(handler_module)
        name = mod.AUTH_RESOURCE
        for handler in mod.default_handlers:
            url_regex = handler[0]
            resource_map[url_regex] = name
    for url_regex in ['/terminals/websocket/(\\w+)', '/api/terminals', '/api/terminals/(\\w+)']:
        resource_map[url_regex] = 'terminals'
    return resource_map