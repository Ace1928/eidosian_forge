from ..type.definition import (GraphQLInterfaceType, GraphQLList,
def is_equal_type(type_a, type_b):
    if type_a is type_b:
        return True
    if isinstance(type_a, GraphQLNonNull) and isinstance(type_b, GraphQLNonNull):
        return is_equal_type(type_a.of_type, type_b.of_type)
    if isinstance(type_a, GraphQLList) and isinstance(type_b, GraphQLList):
        return is_equal_type(type_a.of_type, type_b.of_type)
    return False