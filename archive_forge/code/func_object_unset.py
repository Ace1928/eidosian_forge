import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def object_unset(self, container, object, properties):
    """Unset object properties

        :param string container:
            container name for object to modify
        :param string object:
            name of object to modify
        :param dict properties:
            properties to remove from the object
        """
    headers = self._unset_properties(properties, 'X-Remove-Object-Meta-%s')
    if headers:
        self.create('%s/%s' % (urllib.parse.quote(container), urllib.parse.quote(object)), headers=headers)