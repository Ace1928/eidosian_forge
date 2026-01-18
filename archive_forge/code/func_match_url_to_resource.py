import importlib
import random
import re
import warnings
def match_url_to_resource(url, regex_mapping=None):
    """Finds the JupyterHandler regex pattern that would
    match the given URL and returns the resource name (str)
    of that handler.

    e.g.
    /api/contents/... returns "contents"
    """
    if not regex_mapping:
        regex_mapping = get_regex_to_resource_map()
    for regex, auth_resource in regex_mapping.items():
        pattern = re.compile(regex)
        if pattern.fullmatch(url):
            return auth_resource