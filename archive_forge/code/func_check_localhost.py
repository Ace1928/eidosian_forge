import os
import warnings
import builtins
import cherrypy
def check_localhost(self):
    """Warn if any socket_host is 'localhost'. See #711."""
    for k, v in cherrypy.config.items():
        if k == 'server.socket_host' and v == 'localhost':
            warnings.warn("The use of 'localhost' as a socket host can cause problems on newer systems, since 'localhost' can map to either an IPv4 or an IPv6 address. You should use '127.0.0.1' or '[::1]' instead.")