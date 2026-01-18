from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
def bind_entrypoint(self, ep: Entrypoint):
    ep.bind_dependency_overrides_provider(self)
    self.routes.extend(ep.routes)
    self.on_event('shutdown')(ep.shutdown)