import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
def extract_grouped_output_callback_args(args, kwargs):
    if 'output' in kwargs:
        parameters = kwargs['output']
        if isinstance(parameters, (list, tuple)):
            parameters = list(parameters)
        for dep in flatten_grouping(parameters):
            if not isinstance(dep, Output):
                raise ValueError(f'Invalid value provided where an Output dependency object was expected: {dep}')
        return parameters
    parameters = []
    while args:
        next_deps = flatten_grouping(args[0])
        if all((isinstance(d, Output) for d in next_deps)):
            parameters.append(args.pop(0))
        else:
            break
    return parameters