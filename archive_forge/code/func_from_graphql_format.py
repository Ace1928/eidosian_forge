import typing
from re import sub
def from_graphql_format(data: typing.Dict) -> typing.Dict:
    """
    converts all camelcase keys in the data to
    snake_case
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[to_snake_case(key)] = from_graphql_format(value)
        elif isinstance(value, list):
            values = []
            for item in value:
                if isinstance(item, (dict, list)):
                    values.append(from_graphql_format(item))
                else:
                    values.append(item)
            result[to_snake_case(key)] = values
        else:
            result[to_snake_case(key)] = value
    return result