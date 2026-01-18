from keystone.common import resource_options
from keystone.common.resource_options import options as ro_opt
def register_role_options():
    for opt in [ro_opt.IMMUTABLE_OPT]:
        PROJECT_OPTIONS_REGISTRY.register_option(opt)