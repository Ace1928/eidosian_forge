from keystone.common import validation
from keystone.i18n import _
@property
def option_names(self):
    return set([opt.option_name for opt in self.options])