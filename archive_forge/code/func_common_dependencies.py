from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
@property
def common_dependencies(self):
    return self.entrypoint_route.common_dependencies