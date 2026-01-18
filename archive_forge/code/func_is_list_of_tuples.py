import sys
def is_list_of_tuples(value):
    if not value or not isinstance(value, (list, tuple)) or (not all((isinstance(t, tuple) and len(t) == 2 for t in value))):
        return (False, None)
    return (True, value)