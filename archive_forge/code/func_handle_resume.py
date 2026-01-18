import random
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_utils import timeutils
def handle_resume(self):
    return self._handle_action()