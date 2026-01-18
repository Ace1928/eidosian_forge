import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
def handle_grouped_callback_args(args, kwargs):
    """Split args into outputs, inputs and states"""
    prevent_initial_call = kwargs.get('prevent_initial_call', None)
    if prevent_initial_call is None and args and isinstance(args[-1], bool):
        args, prevent_initial_call = (args[:-1], args[-1])
    flat_args = []
    for arg in args:
        flat_args += arg if isinstance(arg, (list, tuple)) else [arg]
    outputs = extract_grouped_output_callback_args(flat_args, kwargs)
    flat_outputs = flatten_grouping(outputs)
    if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
        out0 = kwargs.get('output', args[0] if args else None)
        if not isinstance(out0, (list, tuple)):
            outputs = outputs[0]
    inputs_state = extract_grouped_input_state_callback_args(flat_args, kwargs)
    flat_inputs, flat_state, input_state_indices = compute_input_state_grouping_indices(inputs_state)
    types = (Input, Output, State)
    validate_callback(flat_outputs, flat_inputs, flat_state, flat_args, types)
    return (outputs, flat_inputs, flat_state, input_state_indices, prevent_initial_call)