from cheroot.wsgi import PathInfoDispatcher
def test_dispatch_no_script_name():
    """Dispatch despite lack of ``SCRIPT_NAME`` in environ."""

    def app(environ, start_response):
        start_response('200 OK', [('Content-Type', 'text/plain; charset=utf-8')])
        return [u'Hello, world!'.encode('utf-8')]
    d = PathInfoDispatcher([('/', app)])
    response = wsgi_invoke(d, {'PATH_INFO': '/foo'})
    assert response == {'status': '200 OK', 'headers': [('Content-Type', 'text/plain; charset=utf-8')], 'body': b'Hello, world!'}