import os as _os
import re as _re
import sys as _sys
import warnings
from gettext import gettext as _, ngettext
def parse_known_intermixed_args(self, args=None, namespace=None):
    positionals = self._get_positional_actions()
    a = [action for action in positionals if action.nargs in [PARSER, REMAINDER]]
    if a:
        raise TypeError('parse_intermixed_args: positional arg with nargs=%s' % a[0].nargs)
    if [action.dest for group in self._mutually_exclusive_groups for action in group._group_actions if action in positionals]:
        raise TypeError('parse_intermixed_args: positional in mutuallyExclusiveGroup')
    try:
        save_usage = self.usage
        try:
            if self.usage is None:
                self.usage = self.format_usage()[7:]
            for action in positionals:
                action.save_nargs = action.nargs
                action.nargs = SUPPRESS
                action.save_default = action.default
                action.default = SUPPRESS
            namespace, remaining_args = self.parse_known_args(args, namespace)
            for action in positionals:
                if hasattr(namespace, action.dest) and getattr(namespace, action.dest) == []:
                    from warnings import warn
                    warn('Do not expect %s in %s' % (action.dest, namespace))
                    delattr(namespace, action.dest)
        finally:
            for action in positionals:
                action.nargs = action.save_nargs
                action.default = action.save_default
        optionals = self._get_optional_actions()
        try:
            for action in optionals:
                action.save_required = action.required
                action.required = False
            for group in self._mutually_exclusive_groups:
                group.save_required = group.required
                group.required = False
            namespace, extras = self.parse_known_args(remaining_args, namespace)
        finally:
            for action in optionals:
                action.required = action.save_required
            for group in self._mutually_exclusive_groups:
                group.required = group.save_required
    finally:
        self.usage = save_usage
    return (namespace, extras)