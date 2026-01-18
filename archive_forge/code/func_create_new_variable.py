import re
def create_new_variable(next_creator, **kwargs):
    """Create the variable using `next_creator` and store it."""
    canonical_name = _canonicalize_variable_name(kwargs.get('name'))
    v = next_creator(**kwargs)
    if canonical_name not in shared_variable_store:
        shared_variable_store[canonical_name] = []
    shared_variable_store[canonical_name].append(v)
    return v