import re
from pandocfilters import RawInline, applyJSONFilters, stringify  # type:ignore[import-untyped]
def resolve_one_reference(key, val, fmt, meta):
    """
    This takes a tuple of arguments that are compatible with ``pandocfilters.walk()`` that
    allows identifying hyperlinks in the document and transforms them into valid LaTeX
    \\hyperref{} calls so that linking to headers between cells is possible.

    See the documentation in ``pandocfilters.walk()`` for further information on the meaning
    and specification of ``key``, ``val``, ``fmt``, and ``meta``.
    """
    if key == 'Link':
        text = stringify(val[1])
        target = val[2][0]
        m = re.match('#(.+)$', target)
        if m:
            label = m.group(1).lower()
            label = re.sub('[^\\w-]+', '', label)
            text = re.sub('_', '\\_', text)
            return RawInline('tex', f'\\hyperref[{label}]{{{text}}}')
    return None