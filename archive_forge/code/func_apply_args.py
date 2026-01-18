from __future__ import annotations
import argparse
import dataclasses
from typing import (
from typing_extensions import Annotated, get_args, get_origin
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
def apply_args(self, parser: argparse.ArgumentParser, parent: Optional[ParserSpecification]=None) -> None:
    """Create defined arguments and subparsers."""

    def format_group_name(prefix: str) -> str:
        return (prefix + ' options').strip()
    group_from_prefix: Dict[str, argparse._ArgumentGroup] = {'': parser._action_groups[1], **{cast(str, group.title).partition(' ')[0]: group for group in parser._action_groups[2:]}}
    positional_group = parser._action_groups[0]
    assert positional_group.title == 'positional arguments'
    for arg in self.args:
        if arg.lowered.help is not argparse.SUPPRESS and arg.extern_prefix not in group_from_prefix:
            description = parent.helptext_from_intern_prefixed_field_name.get(arg.intern_prefix) if parent is not None else None
            group_from_prefix[arg.extern_prefix] = parser.add_argument_group(format_group_name(arg.extern_prefix), description=description)
    for arg in self.args:
        if arg.field.is_positional():
            arg.add_argument(positional_group)
            continue
        if arg.extern_prefix in group_from_prefix:
            arg.add_argument(group_from_prefix[arg.extern_prefix])
        else:
            assert arg.lowered.help is argparse.SUPPRESS
            arg.add_argument(group_from_prefix[''])
    for child in self.child_from_prefix.values():
        child.apply_args(parser, parent=self)