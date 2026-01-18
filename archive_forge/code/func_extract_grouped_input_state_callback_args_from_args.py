import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
def extract_grouped_input_state_callback_args_from_args(args):
    parameters = []
    while args:
        next_deps = flatten_grouping(args[0])
        if all((isinstance(d, (Input, State)) for d in next_deps)):
            parameters.append(args.pop(0))
        else:
            break
    if len(parameters) == 1:
        return parameters[0]
    return parameters