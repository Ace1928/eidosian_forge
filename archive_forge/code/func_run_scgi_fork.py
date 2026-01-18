from paste.deploy.converters import aslist, asbool
from paste.script.serve import ensure_port_cleanup
import warnings
def run_scgi_fork(wsgi_app, global_conf, scriptName='', host='localhost', port='4000', allowedServers='127.0.0.1'):
    import flup.server.scgi_fork
    warn('scgi_fork')
    addr = (host, int(port))
    ensure_port_cleanup([addr])
    s = flup.server.scgi_fork.WSGIServer(wsgi_app, scriptName=scriptName, bindAddress=addr, allowedServers=aslist(allowedServers))
    s.run()