from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
@raw_response.setter
def raw_response(self, value: dict):
    self.on_raw_response(value)