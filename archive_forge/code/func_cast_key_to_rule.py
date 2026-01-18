import functools
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import function
def cast_key_to_rule(self, key):
    return '.'.join([item for item in key.split('.') if not item.isdigit()])