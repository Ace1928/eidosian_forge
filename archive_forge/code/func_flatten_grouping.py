from dash.exceptions import InvalidCallbackReturnValue
from ._utils import AttributeDict, stringify_id
def flatten_grouping(grouping, schema=None):
    """
    Convert a grouping value to a list of scalar values

    :param grouping: grouping value to flatten
    :param schema: If provided, a grouping value representing the expected structure of
        the input grouping value. If not provided, the grouping value is its own schema.
        A schema is required in order to be able treat tuples and dicts in the input
        grouping as scalar values.

    :return: list of the scalar values in the input grouping
    """
    if schema is None:
        schema = grouping
    else:
        validate_grouping(grouping, schema)
    if isinstance(schema, (tuple, list)):
        return [g for group_el, schema_el in zip(grouping, schema) for g in flatten_grouping(group_el, schema_el)]
    if isinstance(schema, dict):
        return [g for k in schema for g in flatten_grouping(grouping[k], schema[k])]
    return [grouping]