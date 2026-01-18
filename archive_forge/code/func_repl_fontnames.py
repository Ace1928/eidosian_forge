import io
import math
import os
import typing
import weakref
def repl_fontnames(doc):
    """Populate 'font_buffers'.

        For each font candidate, store its xref and the list of names
        by which PDF text may refer to it (there may be multiple).
        """

    def norm_name(name):
        """Recreate font name that contains PDF hex codes.

            E.g. #20 -> space, chr(32)
            """
        while '#' in name:
            p = name.find('#')
            c = int(name[p + 1:p + 3], 16)
            name = name.replace(name[p:p + 3], chr(c))
        return name

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
    for i in range(doc.page_count):
        for f in doc.get_page_fonts(i, full=True):
            font_xref = f[0]
            font_ext = f[1]
            basename = f[3]
            if font_ext not in ('otf', 'ttf', 'woff', 'woff2'):
                continue
            if len(basename) > 6 and basename[6] == '+':
                continue
            extr = doc.extract_font(font_xref)
            fontbuffer = extr[-1]
            names = get_fontnames(doc, f)
            name_set, xref_set, subsets = font_buffers.get(fontbuffer, (set(), set(), (set(), set())))
            xref_set.add(font_xref)
            for name in names:
                name_set.add(name)
            font = fitz.Font(fontbuffer=fontbuffer)
            name_set.add(font.name)
            del font
            font_buffers[fontbuffer] = (name_set, xref_set, subsets)