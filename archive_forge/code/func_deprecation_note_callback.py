from the :func:`setup()` function.
import logging
import types
import docutils.nodes
import docutils.utils
from humanfriendly.deprecation import get_aliases
from humanfriendly.text import compact, dedent, format
from humanfriendly.usage import USAGE_MARKER, render_usage
def deprecation_note_callback(app, what, name, obj, options, lines):
    """
    Automatically document aliases defined using :func:`~humanfriendly.deprecation.define_aliases()`.

    Refer to :func:`enable_deprecation_notes()` to enable the use of this
    function (you probably don't want to call :func:`deprecation_note_callback()`
    directly).

    This function implements a callback for ``autodoc-process-docstring`` that
    reformats module docstrings to append an overview of aliases defined by the
    module.

    The parameters expected by this function are those defined for Sphinx event
    callback functions (i.e. I'm not going to document them here :-).
    """
    if isinstance(obj, types.ModuleType) and lines:
        aliases = get_aliases(obj.__name__)
        if aliases:
            blocks = [dedent('\n'.join(lines))]
            blocks.append('.. note:: Deprecated names')
            indent = ' ' * 3
            if len(aliases) == 1:
                explanation = '\n                    The following alias exists to preserve backwards compatibility,\n                    however a :exc:`~exceptions.DeprecationWarning` is triggered\n                    when it is accessed, because this alias will be removed\n                    in a future release.\n                '
            else:
                explanation = '\n                    The following aliases exist to preserve backwards compatibility,\n                    however a :exc:`~exceptions.DeprecationWarning` is triggered\n                    when they are accessed, because these aliases will be\n                    removed in a future release.\n                '
            blocks.append(indent + compact(explanation))
            for name, target in aliases.items():
                blocks.append(format('%s.. data:: %s', indent, name))
                blocks.append(format('%sAlias for :obj:`%s`.', indent * 2, target))
            update_lines(lines, '\n\n'.join(blocks))