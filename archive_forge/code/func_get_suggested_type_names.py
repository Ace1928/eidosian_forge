from collections import Counter
from ...error import GraphQLError
from ...pyutils.ordereddict import OrderedDict
from ...type.definition import (GraphQLInterfaceType, GraphQLObjectType,
from ...utils.quoted_or_list import quoted_or_list
from ...utils.suggestion_list import suggestion_list
from .base import ValidationRule
def get_suggested_type_names(schema, output_type, field_name):
    """Go through all of the implementations of type, as well as the interfaces
      that they implement. If any of those types include the provided field,
      suggest them, sorted by how often the type is referenced,  starting
      with Interfaces."""
    if isinstance(output_type, (GraphQLInterfaceType, GraphQLUnionType)):
        suggested_object_types = []
        interface_usage_count = OrderedDict()
        for possible_type in schema.get_possible_types(output_type):
            if not possible_type.fields.get(field_name):
                return
            suggested_object_types.append(possible_type.name)
            for possible_interface in possible_type.interfaces:
                if not possible_interface.fields.get(field_name):
                    continue
                interface_usage_count[possible_interface.name] = interface_usage_count.get(possible_interface.name, 0) + 1
        suggested_interface_types = sorted(list(interface_usage_count.keys()), key=lambda k: interface_usage_count[k], reverse=True)
        suggested_interface_types.extend(suggested_object_types)
        return suggested_interface_types
    return []