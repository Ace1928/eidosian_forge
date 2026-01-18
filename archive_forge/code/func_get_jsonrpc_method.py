from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
def get_jsonrpc_method() -> Optional[str]:
    return get_jsonrpc_context().raw_request.get('method')