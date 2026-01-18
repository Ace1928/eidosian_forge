import os
import socket
import time
from oslo_serialization import jsonutils
import tenacity
from octavia_lib.api.drivers import data_models
from octavia_lib.api.drivers import exceptions as driver_exceptions
from octavia_lib.common import constants
def get_healthmonitor(self, healthmonitor_id):
    """Get a health monitor object.

        :param healthmonitor_id: The health monitor ID to lookup.
        :type healthmonitor_id: UUID string
        :raises DriverAgentTimeout: The driver agent did not respond
          inside the timeout.
        :raises DriverError: An unexpected error occurred.
        :returns: A HealthMonitor object or None if not found.
        """
    data = self._get_resource(constants.HEALTHMONITORS, healthmonitor_id)
    if data:
        return data_models.HealthMonitor.from_dict(data)
    return None