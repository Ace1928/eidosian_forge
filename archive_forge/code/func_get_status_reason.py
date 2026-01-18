from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from urllib import parse
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients.os import swift
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def get_status_reason(self, status):
    return [s[self.REASON] for s in self.get_signals() if s[self.STATUS] == status]