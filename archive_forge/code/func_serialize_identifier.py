def serialize_identifier(value):
    """Serialize any string as a CSS identifier

    :type value: :obj:`str`
    :param value: A string representing a CSS value.
    :returns:
        A :obj:`string <str>` that would parse as an
        :class:`tinycss2.ast.IdentToken` whose
        :attr:`tinycss2.ast.IdentToken.value` attribute equals the passed
        ``value`` argument.

    """
    if value == '-':
        return '\\-'
    if value[:2] == '--':
        return '--' + serialize_name(value[2:])
    if value[0] == '-':
        result = '-'
        value = value[1:]
    else:
        result = ''
    c = value[0]
    result += c if c in 'abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ' or ord(c) > 127 else '\\A ' if c == '\n' else '\\D ' if c == '\r' else '\\C ' if c == '\x0c' else '\\%X ' % ord(c) if c in '0123456789' else '\\' + c
    result += serialize_name(value[1:])
    return result