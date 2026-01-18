from io import IOBase
from django.conf import settings
from django.core import signals
from django.core.handlers import base
from django.http import HttpRequest, QueryDict, parse_cookie
from django.urls import set_script_prefix
from django.utils.encoding import repercent_broken_unicode
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
def get_script_name(environ):
    """
    Return the equivalent of the HTTP request's SCRIPT_NAME environment
    variable. If Apache mod_rewrite is used, return what would have been
    the script name prior to any rewriting (so it's the script name as seen
    from the client's perspective), unless the FORCE_SCRIPT_NAME setting is
    set (to anything).
    """
    if settings.FORCE_SCRIPT_NAME is not None:
        return settings.FORCE_SCRIPT_NAME
    script_url = get_bytes_from_wsgi(environ, 'SCRIPT_URL', '') or get_bytes_from_wsgi(environ, 'REDIRECT_URL', '')
    if script_url:
        if b'//' in script_url:
            script_url = _slashes_re.sub(b'/', script_url)
        path_info = get_bytes_from_wsgi(environ, 'PATH_INFO', '')
        script_name = script_url.removesuffix(path_info)
    else:
        script_name = get_bytes_from_wsgi(environ, 'SCRIPT_NAME', '')
    return script_name.decode()