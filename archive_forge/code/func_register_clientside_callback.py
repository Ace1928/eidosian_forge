import collections
import hashlib
from functools import wraps
import flask
from .dependencies import (
from .exceptions import (
from ._grouping import (
from ._utils import (
from . import _validate
from .long_callback.managers import BaseLongCallbackManager
from ._callback_context import context_value
def register_clientside_callback(callback_list, callback_map, config_prevent_initial_callbacks, inline_scripts, clientside_function, *args, **kwargs):
    output, inputs, state, prevent_initial_call = handle_callback_args(args, kwargs)
    insert_callback(callback_list, callback_map, config_prevent_initial_callbacks, output, None, inputs, state, None, prevent_initial_call)
    if isinstance(clientside_function, str):
        namespace = '_dashprivate_clientside_funcs'
        function_name = hashlib.md5(clientside_function.encode('utf-8')).hexdigest()
        inline_scripts.append(_inline_clientside_template.format(namespace=namespace, function_name=function_name, clientside_function=clientside_function))
    else:
        namespace = clientside_function.namespace
        function_name = clientside_function.function_name
    callback_list[-1]['clientside_function'] = {'namespace': namespace, 'function_name': function_name}