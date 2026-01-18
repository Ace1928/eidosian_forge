from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
def bind_dependency_overrides_provider(self, value):
    for route in self.routes:
        route.dependency_overrides_provider = value