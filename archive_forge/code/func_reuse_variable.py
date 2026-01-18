import re
def reuse_variable(next_creator, **kwargs):
    """Re-use existing variable from store with same name (in order)."""
    del next_creator
    name = kwargs.get('name')
    canonical_name = _canonicalize_variable_name(name)
    try:
        variable_index = variable_scope_access_index.get(canonical_name, 0)
        v = shared_variable_store[canonical_name][variable_index]
        variable_scope_access_index[canonical_name] = variable_index + 1
        return v
    except (KeyError, IndexError):
        raise RuntimeError('Tried to create variable {} with mismatching name on device {}'.format(name, device_id))