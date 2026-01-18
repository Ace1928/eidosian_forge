def parse_datatypes(f):
    dt = set()
    for line in f:
        if '<sect1' in line:
            break
        if '<entry><type>' not in line:
            continue
        line = re.sub('<replaceable>[^<]+</replaceable>', '', line)
        line = re.sub('<[^>]+>', '', line)
        for tmp in [t for tmp in line.split('[') for t in tmp.split(']') if '(' not in t]:
            for t in tmp.split(','):
                t = t.strip()
                if not t:
                    continue
                dt.add(' '.join(t.split()))
    dt = list(dt)
    dt.sort()
    return dt