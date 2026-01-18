import collections
import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib.i18n import _
from osc_lib import utils
from oslo_utils import excutils
from osc_placement.resources import common
from osc_placement import version
def parse_resource_argument(resource):
    parts = resource.split('=')
    if len(parts) != 2:
        raise ValueError('Resource argument must have "name=value" format')
    name, value = parts
    parts = name.split(':')
    if len(parts) == 2:
        name, field = parts
    elif len(parts) == 1:
        name = parts[0]
        field = 'total'
    else:
        raise ValueError('Resource argument can contain only one colon')
    if not all([name, field, value]):
        raise ValueError('Name, field and value must be not empty')
    if field not in INVENTORY_FIELDS:
        raise ValueError('Unknown inventory field %s' % field)
    value = INVENTORY_FIELDS[field]['type'](value)
    return (name, field, value)