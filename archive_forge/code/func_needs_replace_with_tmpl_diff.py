from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import function
from heat.engine import properties
from heat.engine import resource
def needs_replace_with_tmpl_diff(self, tmpl_diff):
    return tmpl_diff.metadata_changed()