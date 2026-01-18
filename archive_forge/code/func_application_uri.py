import posixpath
def application_uri(environ):
    """Return the application's base URI (no PATH_INFO or QUERY_STRING)"""
    url = environ['wsgi.url_scheme'] + '://'
    from urllib.parse import quote
    if environ.get('HTTP_HOST'):
        url += environ['HTTP_HOST']
    else:
        url += environ['SERVER_NAME']
        if environ['wsgi.url_scheme'] == 'https':
            if environ['SERVER_PORT'] != '443':
                url += ':' + environ['SERVER_PORT']
        elif environ['SERVER_PORT'] != '80':
            url += ':' + environ['SERVER_PORT']
    url += quote(environ.get('SCRIPT_NAME') or '/', encoding='latin1')
    return url