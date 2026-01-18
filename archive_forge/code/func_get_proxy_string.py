import platform
import six
from blessed.colorspace import CGA_COLORS, X11_COLORNAMES_TO_RGB
def get_proxy_string(term, attr):
    """
    Proxy and return callable string for proxied attributes.

    :arg Terminal term: :class:`~.Terminal` instance.
    :arg str attr: terminal capability name that may be proxied.
    :rtype: None or :class:`ParameterizingProxyString`.
    :returns: :class:`ParameterizingProxyString` for some attributes
        of some terminal types that support it, where the terminfo(5)
        database would otherwise come up empty, such as ``move_x``
        attribute for ``term.kind`` of ``screen``.  Otherwise, None.
    """
    term_kind = next(iter((_kind for _kind in ('screen', 'ansi') if term.kind.startswith(_kind))), term)
    _proxy_table = {'screen': {'hpa': ParameterizingProxyString((u'\x1b[{0}G', lambda *arg: (arg[0] + 1,)), term.normal, attr), 'vpa': ParameterizingProxyString((u'\x1b[{0}d', lambda *arg: (arg[0] + 1,)), term.normal, attr)}, 'ansi': {'civis': ParameterizingProxyString((u'\x1b[?25l', lambda *arg: ()), term.normal, attr), 'cnorm': ParameterizingProxyString((u'\x1b[?25h', lambda *arg: ()), term.normal, attr), 'hpa': ParameterizingProxyString((u'\x1b[{0}G', lambda *arg: (arg[0] + 1,)), term.normal, attr), 'vpa': ParameterizingProxyString((u'\x1b[{0}d', lambda *arg: (arg[0] + 1,)), term.normal, attr), 'sc': '\x1b[s', 'rc': '\x1b[u'}}
    return _proxy_table.get(term_kind, {}).get(attr, None)