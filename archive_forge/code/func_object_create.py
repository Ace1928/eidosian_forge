import configparser
import os
import time
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
def object_create(self, object_name, params):
    """Create an object.

        :param object_name: object name
        :param params: parameters to cinder command
        :return: object dictionary
        """
    cmd = self.object_cmd(object_name, 'create')
    output = self.cinder(cmd, params=params)
    object = self._get_property_from_output(output)
    self.addCleanup(self.object_delete, object_name, object['id'])
    if object_name in ('volume', 'snapshot', 'backup'):
        self.wait_for_object_status(object_name, object['id'], 'available')
    return object