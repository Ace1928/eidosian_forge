def pprettyprint(parsedxml):
    """pretty printer mainly for testing"""
    if isinstance(parsedxml, (str, bytes)):
        return parsedxml
    name, attdict, textlist, extra = parsedxml
    if not attdict:
        attdict = {}
    attlist = []
    for k in attdict.keys():
        v = attdict[k]
        attlist.append('%s=%s' % (k, repr(v)))
    attributes = ' '.join(attlist)
    if not name and attributes:
        raise ValueError('name missing with attributes???')
    if textlist is not None:
        textlistpprint = list(map(pprettyprint, textlist))
        textpprint = '\n'.join(textlistpprint)
        if not name:
            return textpprint
        nllist = textpprint.split('\n')
        textpprint = '   ' + '\n   '.join(nllist)
        return '<%s %s>\n%s\n</%s>' % (name, attributes, textpprint, name)
    return '<%s %s/>' % (name, attributes)