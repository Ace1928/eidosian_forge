import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def object_set(self, container, object, properties):
    """Set object properties

        :param string container:
            container name for object to modify
        :param string object:
            name of object to modify
        :param dict properties:
            properties to add or update for the container
        """
    headers = self._set_properties(properties, 'X-Object-Meta-%s')
    if headers:
        self.create('%s/%s' % (urllib.parse.quote(container), urllib.parse.quote(object)), headers=headers)