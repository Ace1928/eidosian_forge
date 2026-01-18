from typing import Any, List
def stringify_value(val: Any) -> str:
    """Stringify a value.

    Args:
        val: The value to stringify.

    Returns:
        str: The stringified value.
    """
    if isinstance(val, str):
        return val
    elif isinstance(val, dict):
        return '\n' + stringify_dict(val)
    elif isinstance(val, list):
        return '\n'.join((stringify_value(v) for v in val))
    else:
        return str(val)