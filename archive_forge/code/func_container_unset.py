import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def container_unset(self, container, properties):
    """Unset container properties

        :param string container:
            name of container to modify
        :param dict properties:
            properties to remove from the container
        """
    headers = self._unset_properties(properties, 'X-Remove-Container-Meta-%s')
    if headers:
        self.create(urllib.parse.quote(container), headers=headers)