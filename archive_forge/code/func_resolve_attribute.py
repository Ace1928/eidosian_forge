import platform
import six
from blessed.colorspace import CGA_COLORS, X11_COLORNAMES_TO_RGB
def resolve_attribute(term, attr):
    """
    Resolve a terminal attribute name into a capability class.

    :arg Terminal term: :class:`~.Terminal` instance.
    :arg str attr: Sugary, ordinary, or compound formatted terminal
        capability, such as "red_on_white", "normal", "red", or
        "bold_on_black".
    :returns: a string class instance which emits the terminal sequence
        for the given terminal capability, or may be used as a callable to
        wrap the given string with such sequence.
    :returns: :class:`NullCallableString` when
        :attr:`~.Terminal.number_of_colors` is 0,
        otherwise :class:`FormattingString`.
    :rtype: :class:`NullCallableString` or :class:`FormattingString`
    """
    if attr in COLORS:
        return resolve_color(term, attr)
    if attr in COMPOUNDABLES:
        sequence = resolve_capability(term, attr)
        return FormattingString(sequence, term.normal)
    formatters = split_compound(attr)
    if all((fmt in COLORS or fmt in COMPOUNDABLES for fmt in formatters)):
        resolution = (resolve_attribute(term, fmt) for fmt in formatters)
        return FormattingString(u''.join(resolution), term.normal)
    tparm_capseq = resolve_capability(term, attr)
    if not tparm_capseq:
        proxy = get_proxy_string(term, term._sugar.get(attr, attr))
        if proxy is not None:
            return proxy
    return ParameterizingString(tparm_capseq, term.normal, attr)