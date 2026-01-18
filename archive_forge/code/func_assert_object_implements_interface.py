from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from functools import reduce
from ..utils.type_comparators import is_equal_type, is_type_sub_type_of
from .definition import (GraphQLArgument, GraphQLField,
@classmethod
def assert_object_implements_interface(cls, schema, object, interface):
    object_field_map = object.fields
    interface_field_map = interface.fields
    for field_name, interface_field in interface_field_map.items():
        object_field = object_field_map.get(field_name)
        assert object_field, '"{}" expects field "{}" but "{}" does not provide it.'.format(interface, field_name, object)
        assert is_type_sub_type_of(schema, object_field.type, interface_field.type), '{}.{} expects type "{}" but {}.{} provides type "{}".'.format(interface, field_name, interface_field.type, object, field_name, object_field.type)
        for arg_name, interface_arg in interface_field.args.items():
            object_arg = object_field.args.get(arg_name)
            assert object_arg, '{}.{} expects argument "{}" but {}.{} does not provide it.'.format(interface, field_name, arg_name, object, field_name)
            assert is_equal_type(interface_arg.type, object_arg.type), '{}.{}({}:) expects type "{}" but {}.{}({}:) provides type "{}".'.format(interface, field_name, arg_name, interface_arg.type, object, field_name, arg_name, object_arg.type)
        for arg_name, object_arg in object_field.args.items():
            interface_arg = interface_field.args.get(arg_name)
            if not interface_arg:
                assert not isinstance(object_arg.type, GraphQLNonNull), '{}.{}({}:) is of required type "{}" but is not also provided by the interface {}.{}.'.format(object, field_name, arg_name, object_arg.type, interface, field_name)