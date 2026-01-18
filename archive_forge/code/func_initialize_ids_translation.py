from _pydevd_bundle._debug_adapter.pydevd_schema_log import debug_exception
import json
import itertools
from functools import partial
@staticmethod
def initialize_ids_translation():
    BaseSchema._dap_id_to_obj_id = {0: 0, None: None}
    BaseSchema._obj_id_to_dap_id = {0: 0, None: None}
    BaseSchema._next_dap_id = partial(next, itertools.count(1))