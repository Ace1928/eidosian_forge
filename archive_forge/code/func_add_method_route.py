from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
def add_method_route(self, func: FunctionType, *, name: str=None, **kwargs) -> None:
    name = name or func.__name__
    route = self.method_route_class(self, self.entrypoint_route.path + '/' + name, func, name=name, request_class=self.request_class, **kwargs)
    self.routes.append(route)