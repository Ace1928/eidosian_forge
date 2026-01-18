import re
import sys
import warnings
def check_errors(wsgi_errors):
    for attr in ['flush', 'write', 'writelines']:
        assert_(hasattr(wsgi_errors, attr), "wsgi.errors (%r) doesn't have the attribute %s" % (wsgi_errors, attr))