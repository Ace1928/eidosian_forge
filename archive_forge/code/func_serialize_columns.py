from collections import defaultdict
import numpy as np
def serialize_columns(data_set_cols, obj=None):
    if data_set_cols is None:
        return None
    layers = defaultdict(dict)
    length = {}
    for col in data_set_cols:
        accessor_attribute = array_to_binary(col['np_data'])
        if length.get(col['layer_id']):
            length[col['layer_id']] = max(length[col['layer_id']], accessor_attribute['length'])
        else:
            length[col['layer_id']] = accessor_attribute['length']
        if not layers[col['layer_id']].get('attributes'):
            layers[col['layer_id']]['attributes'] = {}
        layers[col['layer_id']]['attributes'][col['accessor']] = {'value': accessor_attribute['value'], 'dtype': accessor_attribute['dtype'], 'size': accessor_attribute['size']}
    for layer_key, _ in layers.items():
        layers[layer_key]['length'] = length[layer_key]
    return layers