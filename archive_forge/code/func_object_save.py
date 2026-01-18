import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def object_save(self, container=None, object=None, file=None):
    """Save an object stored in a container

        :param string container:
            name of container that stores object
        :param string object:
            name of object to save
        :param string file:
            local name of object
        """
    if not file:
        file = object
    response = self._request('GET', '%s/%s' % (urllib.parse.quote(container), urllib.parse.quote(object)), stream=True)
    if response.status_code == 200:
        if file == '-':
            with os.fdopen(sys.stdout.fileno(), 'wb') as f:
                for chunk in response.iter_content(64 * 1024):
                    f.write(chunk)
        else:
            if not os.path.exists(os.path.dirname(file)):
                if len(os.path.dirname(file)) > 0:
                    os.makedirs(os.path.dirname(file))
            with open(file, 'wb') as f:
                for chunk in response.iter_content(64 * 1024):
                    f.write(chunk)