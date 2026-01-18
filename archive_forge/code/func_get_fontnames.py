import io
import math
import os
import typing
import weakref
def get_fontnames(doc, item):
    """Return a list of fontnames for an item of page.get_fonts().

            There may be multiple names e.g. for Type0 fonts.
            """
    fontname = item[3]
    names = [fontname]
    fontname = doc.xref_get_key(item[0], 'BaseFont')[1][1:]
    fontname = norm_name(fontname)
    if fontname not in names:
        names.append(fontname)
    descendents = doc.xref_get_key(item[0], 'DescendantFonts')
    if descendents[0] != 'array':
        return names
    descendents = descendents[1][1:-1]
    if descendents.endswith(' 0 R'):
        xref = int(descendents[:-4])
        descendents = doc.xref_object(xref, compressed=True)
    p1 = descendents.find('/BaseFont')
    if p1 >= 0:
        p2 = descendents.find('/', p1 + 1)
        p1 = min(descendents.find('/', p2 + 1), descendents.find('>>', p2 + 1))
        fontname = descendents[p2 + 1:p1]
        fontname = norm_name(fontname)
        if fontname not in names:
            names.append(fontname)
    return names