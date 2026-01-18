from oslo_log import log as logging
from oslo_serialization import jsonutils
from requests import exceptions
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import template
from heat.rpc import api as rpc_api
@staticmethod
def get_template_file(template_name, allowed_schemes):
    try:
        return urlfetch.get(template_name, allowed_schemes=allowed_schemes)
    except (IOError, exceptions.RequestException) as r_exc:
        args = {'name': template_name, 'exc': str(r_exc)}
        msg = _('Could not fetch remote template "%(name)s": %(exc)s') % args
        raise exception.NotFound(msg_fmt=msg)