import posixpath
def guess_scheme(environ):
    """Return a guess for whether 'wsgi.url_scheme' should be 'http' or 'https'
    """
    if environ.get('HTTPS') in ('yes', 'on', '1'):
        return 'https'
    else:
        return 'http'