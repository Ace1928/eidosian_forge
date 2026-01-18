import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def container_set(self, container, properties):
    """Set container properties

        :param string container:
            name of container to modify
        :param dict properties:
            properties to add or update for the container
        """
    headers = self._set_properties(properties, 'X-Container-Meta-%s')
    if headers:
        self.create(urllib.parse.quote(container), headers=headers)