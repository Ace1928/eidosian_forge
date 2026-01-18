import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def find_gtype(cs, gtype):
    """Gets a group type by name or ID."""
    return utils.find_resource(cs.group_types, gtype)