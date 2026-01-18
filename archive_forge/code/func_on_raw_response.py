from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
def on_raw_response(self, raw_response: dict, exception: Optional[Exception]=None, is_unhandled_exception: bool=False):
    raw_response.pop('id', None)
    if isinstance(self.raw_request, dict) and 'id' in self.raw_request:
        raw_response['id'] = self.raw_request.get('id')
    elif 'error' in raw_response:
        raw_response['id'] = None
    self._raw_response = raw_response
    self.exception = exception
    self.is_unhandled_exception = is_unhandled_exception