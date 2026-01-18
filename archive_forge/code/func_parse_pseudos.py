def parse_pseudos(f):
    dt = []
    re_start = re.compile('\\s*<table id="datatype-pseudotypes-table">')
    re_entry = re.compile('\\s*<entry><type>([^<]+)</></entry>')
    re_end = re.compile('\\s*</table>')
    f = iter(f)
    for line in f:
        if re_start.match(line) is not None:
            break
    else:
        raise ValueError('pseudo datatypes table not found')
    for line in f:
        m = re_entry.match(line)
        if m is not None:
            dt.append(m.group(1))
        if re_end.match(line) is not None:
            break
    else:
        raise ValueError('end of pseudo datatypes table not found')
    if not dt:
        raise ValueError('pseudo datatypes not found')
    return dt