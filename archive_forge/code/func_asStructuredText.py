import zope.interface
def asStructuredText(I, munge=0, rst=False):
    """ Output structured text format.  Note, this will whack any existing
    'structured' format of the text.

    If `rst=True`, then the output will quote all code as inline literals in
    accordance with 'reStructuredText' markup principles.
    """
    if rst:
        inline_literal = lambda s: '``{}``'.format(s)
    else:
        inline_literal = lambda s: s
    r = [inline_literal(I.getName())]
    outp = r.append
    level = 1
    if I.getDoc():
        outp(_justify_and_indent(_trim_doc_string(I.getDoc()), level))
    bases = [base for base in I.__bases__ if base is not zope.interface.Interface]
    if bases:
        outp(_justify_and_indent('This interface extends:', level, munge))
        level += 1
        for b in bases:
            item = 'o %s' % inline_literal(b.getName())
            outp(_justify_and_indent(_trim_doc_string(item), level, munge))
        level -= 1
    namesAndDescriptions = sorted(I.namesAndDescriptions())
    outp(_justify_and_indent('Attributes:', level, munge))
    level += 1
    for name, desc in namesAndDescriptions:
        if not hasattr(desc, 'getSignatureString'):
            item = '{} -- {}'.format(inline_literal(desc.getName()), desc.getDoc() or 'no documentation')
            outp(_justify_and_indent(_trim_doc_string(item), level, munge))
    level -= 1
    outp(_justify_and_indent('Methods:', level, munge))
    level += 1
    for name, desc in namesAndDescriptions:
        if hasattr(desc, 'getSignatureString'):
            _call = '{}{}'.format(desc.getName(), desc.getSignatureString())
            item = '{} -- {}'.format(inline_literal(_call), desc.getDoc() or 'no documentation')
            outp(_justify_and_indent(_trim_doc_string(item), level, munge))
    return '\n\n'.join(r) + '\n\n'