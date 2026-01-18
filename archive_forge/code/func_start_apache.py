import os
import re
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.process import servers
from cherrypy.test import helper
def start_apache(self):
    fcgiconf = CONF_PATH
    if not os.path.isabs(fcgiconf):
        fcgiconf = os.path.join(curdir, fcgiconf)
    with open(fcgiconf, 'wb') as f:
        server = repr(os.path.join(curdir, 'fastcgi.pyc'))[1:-1]
        output = self.template % {'port': self.port, 'root': curdir, 'server': server}
        output = ntob(output.replace('\r\n', '\n'))
        f.write(output)
    result = read_process(APACHE_PATH, '-k start -f %s' % fcgiconf)
    if result:
        print(result)