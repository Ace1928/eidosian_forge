import re
import sys
import warnings
def check_input(wsgi_input):
    for attr in ['read', 'readline', 'readlines', '__iter__']:
        assert_(hasattr(wsgi_input, attr), "wsgi.input (%r) doesn't have the attribute %s" % (wsgi_input, attr))