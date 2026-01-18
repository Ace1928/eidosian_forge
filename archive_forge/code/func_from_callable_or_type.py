from __future__ import annotations
import argparse
import dataclasses
from typing import (
from typing_extensions import Annotated, get_args, get_origin
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
@staticmethod
def from_callable_or_type(f: Callable[..., T], description: Optional[str], parent_classes: Set[Type[Any]], default_instance: Union[T, _fields.PropagatingMissingType, _fields.NonpropagatingMissingType], intern_prefix: str, extern_prefix: str, subcommand_prefix: str='', support_single_arg_types: bool=False) -> ParserSpecification:
    """Create a parser definition from a callable or type."""
    markers = _resolver.unwrap_annotated(f, _markers._Marker)[1]
    consolidate_subcommand_args = _markers.ConsolidateSubcommandArgs in markers
    f, type_from_typevar, field_list = _fields.field_list_from_callable(f=f, default_instance=default_instance, support_single_arg_types=support_single_arg_types)
    for i in range(len(field_list)):
        field_list[i] = dataclasses.replace(field_list[i], markers=field_list[i].markers | set(markers))
    if f in parent_classes and f is not dict:
        raise _instantiators.UnsupportedTypeAnnotationError(f'Found a cyclic dependency with type {f}.')
    parent_classes = parent_classes | {cast(Type, f)}
    has_required_args = False
    args = []
    helptext_from_intern_prefixed_field_name: Dict[str, Optional[str]] = {}
    child_from_prefix: Dict[str, ParserSpecification] = {}
    subparsers = None
    subparsers_from_prefix = {}
    for field in field_list:
        field_out = handle_field(field, type_from_typevar=type_from_typevar, parent_classes=parent_classes, intern_prefix=intern_prefix, extern_prefix=extern_prefix, subcommand_prefix=subcommand_prefix)
        if isinstance(field_out, _arguments.ArgumentDefinition):
            args.append(field_out)
            if field_out.lowered.required:
                has_required_args = True
        elif isinstance(field_out, SubparsersSpecification):
            subparsers_from_prefix[field_out.intern_prefix] = field_out
            subparsers = add_subparsers_to_leaves(subparsers, field_out)
        elif isinstance(field_out, ParserSpecification):
            nested_parser = field_out
            child_from_prefix[field_out.intern_prefix] = nested_parser
            if nested_parser.has_required_args:
                has_required_args = True
            if nested_parser.subparsers is not None:
                subparsers_from_prefix.update(nested_parser.subparsers_from_intern_prefix)
                subparsers = add_subparsers_to_leaves(subparsers, nested_parser.subparsers)
            class_field_name = _strings.make_field_name([field.intern_name])
            if field.helptext is not None:
                helptext_from_intern_prefixed_field_name[class_field_name] = field.helptext
            else:
                helptext_from_intern_prefixed_field_name[class_field_name] = _docstrings.get_callable_description(nested_parser.f)
            if len(nested_parser.args) >= 1 and _markers._OPTIONAL_GROUP in nested_parser.args[0].field.markers:
                current_helptext = helptext_from_intern_prefixed_field_name[class_field_name]
                helptext_from_intern_prefixed_field_name[class_field_name] = ('' if current_helptext is None else current_helptext + '\n\n') + 'Default: ' + str(field.default)
    return ParserSpecification(f=f, description=_strings.remove_single_line_breaks(description if description is not None else _docstrings.get_callable_description(f)), args=args, field_list=field_list, child_from_prefix=child_from_prefix, helptext_from_intern_prefixed_field_name=helptext_from_intern_prefixed_field_name, subparsers=subparsers, subparsers_from_intern_prefix=subparsers_from_prefix, intern_prefix=intern_prefix, extern_prefix=extern_prefix, has_required_args=has_required_args, consolidate_subcommand_args=consolidate_subcommand_args)