import argparse
from typing import (
from . import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .parsing import (
from .utils import (
def as_subcommand_to(command: str, subcommand: str, parser: argparse.ArgumentParser, *, help: Optional[str]=None, aliases: Optional[List[str]]=None) -> Callable[[ArgparseCommandFunc], ArgparseCommandFunc]:
    """
    Tag this method as a subcommand to an existing argparse decorated command.

    :param command: Command Name. Space-delimited subcommands may optionally be specified
    :param subcommand: Subcommand name
    :param parser: argparse Parser for this subcommand
    :param help: Help message for this subcommand which displays in the list of subcommands of the command we are adding to.
                 This is passed as the help argument to ArgumentParser.add_subparser().
    :param aliases: Alternative names for this subcommand. This is passed as the alias argument to
                    ArgumentParser.add_subparser().
    :return: Wrapper function that can receive an argparse.Namespace
    """

    def arg_decorator(func: ArgparseCommandFunc) -> ArgparseCommandFunc:
        _set_parser_prog(parser, command + ' ' + subcommand)
        if parser.description is None and func.__doc__:
            parser.description = func.__doc__
        setattr(func, constants.SUBCMD_ATTR_COMMAND, command)
        setattr(func, constants.CMD_ATTR_ARGPARSER, parser)
        setattr(func, constants.SUBCMD_ATTR_NAME, subcommand)
        add_parser_kwargs: Dict[str, Any] = dict()
        if help is not None:
            add_parser_kwargs['help'] = help
        if aliases:
            add_parser_kwargs['aliases'] = aliases[:]
        setattr(func, constants.SUBCMD_ATTR_ADD_PARSER_KWARGS, add_parser_kwargs)
        return func
    return arg_decorator