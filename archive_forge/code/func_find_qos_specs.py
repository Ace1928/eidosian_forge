import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def find_qos_specs(cs, qos_specs):
    """Gets a qos specs by ID."""
    return utils.find_resource(cs.qos_specs, qos_specs)