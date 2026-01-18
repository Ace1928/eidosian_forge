from ..error import GraphQLError
from ..language import ast
from ..pyutils.default_ordered_dict import DefaultOrderedDict
from ..type.definition import GraphQLInterfaceType, GraphQLUnionType
from ..type.directives import GraphQLIncludeDirective, GraphQLSkipDirective
from ..type.introspection import (SchemaMetaFieldDef, TypeMetaFieldDef,
from ..utils.type_from_ast import type_from_ast
from .values import get_argument_values, get_variable_values
def collect_fields(ctx, runtime_type, selection_set, fields, prev_fragment_names):
    """
    Given a selectionSet, adds all of the fields in that selection to
    the passed in map of fields, and returns it at the end.

    collect_fields requires the "runtime type" of an object. For a field which
    returns and Interface or Union type, the "runtime type" will be the actual
    Object type returned by that field.
    """
    for selection in selection_set.selections:
        directives = selection.directives
        if isinstance(selection, ast.Field):
            if not should_include_node(ctx, directives):
                continue
            name = get_field_entry_key(selection)
            fields[name].append(selection)
        elif isinstance(selection, ast.InlineFragment):
            if not should_include_node(ctx, directives) or not does_fragment_condition_match(ctx, selection, runtime_type):
                continue
            collect_fields(ctx, runtime_type, selection.selection_set, fields, prev_fragment_names)
        elif isinstance(selection, ast.FragmentSpread):
            frag_name = selection.name.value
            if frag_name in prev_fragment_names or not should_include_node(ctx, directives):
                continue
            prev_fragment_names.add(frag_name)
            fragment = ctx.fragments.get(frag_name)
            frag_directives = fragment.directives
            if not fragment or not should_include_node(ctx, frag_directives) or (not does_fragment_condition_match(ctx, fragment, runtime_type)):
                continue
            collect_fields(ctx, runtime_type, fragment.selection_set, fields, prev_fragment_names)
    return fields