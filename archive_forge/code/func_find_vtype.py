import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def find_vtype(cs, vtype):
    """Gets a volume type by name or ID."""
    return utils.find_resource(cs.volume_types, vtype)