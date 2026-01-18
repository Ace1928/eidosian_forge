import os
import re
import six
from six.moves import urllib
from routes import request_config
def find_controllers(dirname, prefix=''):
    """Locate controllers in a directory"""
    controllers = []
    for fname in os.listdir(dirname):
        filename = os.path.join(dirname, fname)
        if os.path.isfile(filename) and re.match('^[^_]{1,1}.*\\.py$', fname):
            controllers.append(prefix + fname[:-3])
        elif os.path.isdir(filename):
            controllers.extend(find_controllers(filename, prefix=prefix + fname + '/'))
    return controllers