import functools
from wandb_promise import Promise, is_thenable, promise_for_dict
from ...pyutils.cached_property import cached_property
from ...pyutils.default_ordered_dict import DefaultOrderedDict
from ...type import (GraphQLInterfaceType, GraphQLList, GraphQLNonNull,
from ..base import ResolveInfo, Undefined, collect_fields, get_field_def
from ..values import get_argument_values
from ...error import GraphQLError
def get_resolvers(context, type, field_asts):
    from .resolver import field_resolver
    subfield_asts = get_subfield_asts(context, type, field_asts)
    for response_name, field_asts in subfield_asts.items():
        field_ast = field_asts[0]
        field_name = field_ast.name.value
        field_def = get_field_def(context and context.schema, type, field_name)
        if not field_def:
            continue
        field_base_type = get_base_type(field_def.type)
        field_fragment = None
        info = ResolveInfo(field_name, field_asts, field_base_type, parent_type=type, schema=context and context.schema, fragments=context and context.fragments, root_value=context and context.root_value, operation=context and context.operation, variable_values=context and context.variable_values)
        if isinstance(field_base_type, GraphQLObjectType):
            field_fragment = Fragment(type=field_base_type, field_asts=field_asts, info=info, context=context)
        elif isinstance(field_base_type, (GraphQLInterfaceType, GraphQLUnionType)):
            field_fragment = AbstractFragment(abstract_type=field_base_type, field_asts=field_asts, info=info, context=context)
        resolver = field_resolver(field_def, exe_context=context, info=info, fragment=field_fragment)
        args = get_argument_values(field_def.args, field_ast.arguments, context and context.variable_values)
        yield (response_name, Field(resolver, args, context and context.context_value, info))