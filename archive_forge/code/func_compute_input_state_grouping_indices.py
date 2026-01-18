import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
def compute_input_state_grouping_indices(input_state_grouping):
    flat_deps = flatten_grouping(input_state_grouping)
    flat_inputs = [dep for dep in flat_deps if isinstance(dep, Input)]
    flat_state = [dep for dep in flat_deps if isinstance(dep, State)]
    total_inputs = len(flat_inputs)
    input_count = 0
    state_count = 0
    flat_inds = []
    for dep in flat_deps:
        if isinstance(dep, Input):
            flat_inds.append(input_count)
            input_count += 1
        else:
            flat_inds.append(total_inputs + state_count)
            state_count += 1
    grouping_inds = make_grouping_by_index(input_state_grouping, flat_inds)
    return (flat_inputs, flat_state, grouping_inds)