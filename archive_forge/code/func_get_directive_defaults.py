from __future__ import absolute_import
import os
from .. import Utils
def get_directive_defaults():
    for old_option in ShouldBeFromDirective.known_directives:
        value = globals().get(old_option.options_name)
        assert old_option.directive_name in _directive_defaults
        if not isinstance(value, ShouldBeFromDirective):
            if old_option.disallow:
                raise RuntimeError("Option '%s' must be set from directive '%s'" % (old_option.option_name, old_option.directive_name))
            else:
                _directive_defaults[old_option.directive_name] = value
    return _directive_defaults