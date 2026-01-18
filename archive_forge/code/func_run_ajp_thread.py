from paste.deploy.converters import aslist, asbool
from paste.script.serve import ensure_port_cleanup
import warnings
def run_ajp_thread(wsgi_app, global_conf, scriptName='', host='localhost', port='8009', allowedServers='127.0.0.1'):
    import flup.server.ajp
    warn('ajp_thread')
    addr = (host, int(port))
    ensure_port_cleanup([addr])
    s = flup.server.ajp.WSGIServer(wsgi_app, scriptName=scriptName, bindAddress=addr, allowedServers=aslist(allowedServers))
    s.run()