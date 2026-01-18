def parse_parameter_list(s, start=0):
    """Parses a semicolon-separated 'parameter=value' list.

    Returns a tuple (parmlist, chars_consumed), where parmlist
    is a list of tuples (parm_name, parm_value).

    The parameter values will be unquoted and unescaped as needed.

    Empty parameters (as in ";;") are skipped, as is insignificant
    white space.  The list returned is kept in the same order as the
    parameters appear in the string.

    """
    pos = start
    parmlist = []
    while pos < len(s):
        while pos < len(s) and s[pos] in LWS:
            pos += 1
        if pos < len(s) and s[pos] == ';':
            pos += 1
            while pos < len(s) and s[pos] in LWS:
                pos += 1
        if pos >= len(s):
            break
        parmname, k = parse_token(s, pos)
        if parmname:
            pos += k
            while pos < len(s) and s[pos] in LWS:
                pos += 1
            if not (pos < len(s) and s[pos] == '='):
                raise ParseError('Expected an "=" after parameter name', s, pos)
            pos += 1
            while pos < len(s) and s[pos] in LWS:
                pos += 1
            parmval, k = parse_token_or_quoted_string(s, pos)
            pos += k
            parmlist.append((parmname, parmval))
        else:
            break
    return (parmlist, pos - start)