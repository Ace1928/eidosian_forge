from _pydevd_bundle._debug_adapter.pydevd_schema_log import debug_exception
import json
import itertools
from functools import partial
def get_response_class(request):
    if request.__class__ == dict:
        return _responses_to_types[request['command']]
    return _responses_to_types[request.command]