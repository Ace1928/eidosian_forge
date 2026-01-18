from oslo_log import log as logging
from oslo_serialization import jsonutils
import tempfile
from heat.common import auth_plugin
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import template
@property
def cacert(self):
    ctx_props = self.properties.get(self.CONTEXT)
    if ctx_props:
        self._cacert = ctx_props[self.CA_CERT]
    return self._cacert