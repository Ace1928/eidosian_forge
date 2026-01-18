import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def add_subparsers(self, **kwargs):
    if self._subparsers is not None:
        self.error(_('cannot have multiple subparser arguments'))
    kwargs.setdefault('parser_class', type(self))
    if 'title' in kwargs or 'description' in kwargs:
        title = _(kwargs.pop('title', 'subcommands'))
        description = _(kwargs.pop('description', None))
        self._subparsers = self.add_argument_group(title, description)
    else:
        self._subparsers = self._positionals
    if kwargs.get('prog') is None:
        formatter = self._get_formatter()
        positionals = self._get_positional_actions()
        groups = self._mutually_exclusive_groups
        formatter.add_usage(self.usage, positionals, groups, '')
        kwargs['prog'] = formatter.format_help().strip()
    parsers_class = self._pop_action_class(kwargs, 'parsers')
    action = parsers_class(option_strings=[], **kwargs)
    self._subparsers._add_action(action)
    return action