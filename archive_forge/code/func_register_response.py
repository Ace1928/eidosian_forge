from _pydevd_bundle._debug_adapter.pydevd_schema_log import debug_exception
import json
import itertools
from functools import partial
def register_response(command):

    def do_register(cls):
        _responses_to_types[command] = cls
        return cls
    return do_register