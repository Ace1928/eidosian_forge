import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
def handle_callback_args(args, kwargs):
    """Split args into outputs, inputs and states"""
    prevent_initial_call = kwargs.get('prevent_initial_call', None)
    if prevent_initial_call is None and args and isinstance(args[-1], bool):
        args, prevent_initial_call = (args[:-1], args[-1])
    flat_args = []
    for arg in args:
        flat_args += arg if isinstance(arg, (list, tuple)) else [arg]
    outputs = extract_callback_args(flat_args, kwargs, 'output', Output)
    validate_outputs = outputs
    if len(outputs) == 1:
        out0 = kwargs.get('output', args[0] if args else None)
        if not isinstance(out0, (list, tuple)):
            outputs = outputs[0]
    inputs = extract_callback_args(flat_args, kwargs, 'inputs', Input)
    states = extract_callback_args(flat_args, kwargs, 'state', State)
    types = (Input, Output, State)
    validate_callback(validate_outputs, inputs, states, flat_args, types)
    return (outputs, inputs, states, prevent_initial_call)