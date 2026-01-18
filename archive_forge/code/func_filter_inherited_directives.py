from __future__ import absolute_import
from .TreeFragment import parse_from_strings, StringParseContext
from . import Symtab
from . import Naming
from . import Code
@staticmethod
def filter_inherited_directives(current_directives):
    """
        Cython utility code should usually only pick up a few directives from the
        environment (those that intentionally control its function) and ignore most
        other compiler directives. This function provides a sensible default list
        of directives to copy.
        """
    from .Options import _directive_defaults
    utility_code_directives = dict(_directive_defaults)
    inherited_directive_names = ('binding', 'always_allow_keywords', 'allow_none_for_extension_args', 'auto_pickle', 'ccomplex', 'c_string_type', 'c_string_encoding', 'optimize.inline_defnode_calls', 'optimize.unpack_method_calls', 'optimize.unpack_method_calls_in_pyinit', 'optimize.use_switch')
    for name in inherited_directive_names:
        if name in current_directives:
            utility_code_directives[name] = current_directives[name]
    return utility_code_directives