def parse_qvalue_accept_list(s, start=0, item_parser=parse_token):
    """Parses any of the Accept-* style headers with quality factors.

    This is a low-level function.  It returns a list of tuples, each like:
       (item, item_parms, qvalue, accept_parms)

    You can pass in a function which parses each of the item strings, or
    accept the default where the items must be simple tokens.  Note that
    your parser should not consume any paramters (past the special "q"
    paramter anyway).

    The item_parms and accept_parms are each lists of (name,value) tuples.

    The qvalue is the quality factor, a number from 0 to 1 inclusive.

    """
    itemlist = []
    pos = start
    if pos >= len(s):
        raise ParseError('Starting position is beyond the end of the string', s, pos)
    item = None
    while pos < len(s):
        item, k = item_parser(s, pos)
        pos += k
        while pos < len(s) and s[pos] in LWS:
            pos += 1
        if pos >= len(s) or s[pos] in ',;':
            itemparms, qvalue, acptparms = ([], None, [])
            if pos < len(s) and s[pos] == ';':
                pos += 1
                while pos < len(s) and s[pos] in LWS:
                    pos += 1
                parmlist, k = parse_parameter_list(s, pos)
                for p, v in parmlist:
                    if p == 'q' and qvalue is None:
                        try:
                            qvalue = float(v)
                        except ValueError:
                            raise ParseError('qvalue must be a floating point number', s, pos)
                        if qvalue < 0 or qvalue > 1:
                            raise ParseError('qvalue must be between 0 and 1, inclusive', s, pos)
                    elif qvalue is None:
                        itemparms.append((p, v))
                    else:
                        acptparms.append((p, v))
                pos += k
            if item:
                if qvalue is None:
                    qvalue = 1
                itemlist.append((item, itemparms, qvalue, acptparms))
                item = None
            while pos < len(s) and s[pos] == ',':
                pos += 1
                while pos < len(s) and s[pos] in LWS:
                    pos += 1
        else:
            break
    return (itemlist, pos - start)