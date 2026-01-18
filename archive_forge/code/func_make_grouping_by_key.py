from dash.exceptions import InvalidCallbackReturnValue
from ._utils import AttributeDict, stringify_id
def make_grouping_by_key(schema, source, default=None):
    """
    Create a grouping from a schema by using the schema's scalar values to look up
    items in the provided source object.

    :param schema: A grouping of potential keys in source
    :param source: Dict-like object to use to look up scalar grouping value using
        scalar grouping values as keys
    :param default: Default scalar value to use if grouping scalar key is not present
        in source
    :return: grouping
    """
    return map_grouping(lambda s: source.get(s, default), schema)