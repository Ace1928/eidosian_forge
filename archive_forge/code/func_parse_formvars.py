import cgi
from collections.abc import MutableMapping as DictMixin
from urllib import parse as urlparse
from urllib.parse import quote, parse_qsl
from http.cookies import SimpleCookie, CookieError
from paste.util.multidict import MultiDict
def parse_formvars(environ, include_get_vars=True, encoding=None, errors=None):
    """Parses the request, returning a MultiDict of form variables.

    If ``include_get_vars`` is true then GET (query string) variables
    will also be folded into the MultiDict.

    All values should be strings, except for file uploads which are
    left as ``FieldStorage`` instances.

    If the request was not a normal form request (e.g., a POST with an
    XML body) then ``environ['wsgi.input']`` won't be read.
    """
    source = environ['wsgi.input']
    if 'paste.parsed_formvars' in environ:
        parsed, check_source = environ['paste.parsed_formvars']
        if check_source == source:
            return parsed
    formvars = MultiDict()
    ct = environ.get('CONTENT_TYPE', '').partition(';')[0].lower()
    use_cgi = ct in ('', 'application/x-www-form-urlencoded', 'multipart/form-data')
    if not environ.get('CONTENT_LENGTH'):
        environ['CONTENT_LENGTH'] = '0'
    if use_cgi:
        old_query_string = environ.get('QUERY_STRING', '')
        environ['QUERY_STRING'] = ''
        inp = environ['wsgi.input']
        kwparms = {}
        if encoding:
            kwparms['encoding'] = encoding
        if errors:
            kwparms['errors'] = errors
        fs = cgi.FieldStorage(fp=inp, environ=environ, keep_blank_values=True, **kwparms)
        environ['QUERY_STRING'] = old_query_string
        if isinstance(fs.value, list):
            for name in fs.keys():
                values = fs[name]
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    if not value.filename:
                        value = value.value
                    formvars.add(name, value)
    environ['paste.parsed_formvars'] = (formvars, source)
    if include_get_vars:
        formvars.update(parse_querystring(environ))
    return formvars