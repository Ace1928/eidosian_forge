from keystone.common import validation
from keystone.i18n import _
@property
def options_by_name(self):
    return {opt.option_name: opt for opt in self._registered_options.values()}