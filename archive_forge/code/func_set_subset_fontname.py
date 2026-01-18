import io
import math
import os
import typing
import weakref
def set_subset_fontname(new_xref):
    """Generate a name prefix to tag a font as subset.

        We use a random generator to select 6 upper case ASCII characters.
        The prefixed name must be put in the font xref as the "/BaseFont" value
        and in the FontDescriptor object as the '/FontName' value.
        """
    import random
    import string
    prefix = ''.join(random.choices(tuple(string.ascii_uppercase), k=6)) + '+'
    font_str = doc.xref_object(new_xref, compressed=True)
    font_str = font_str.replace('/BaseFont/', '/BaseFont/' + prefix)
    df = doc.xref_get_key(new_xref, 'DescendantFonts')
    if df[0] == 'array':
        df_xref = int(df[1][1:-1].replace('0 R', ''))
        fd = doc.xref_get_key(df_xref, 'FontDescriptor')
        if fd[0] == 'xref':
            fd_xref = int(fd[1].replace('0 R', ''))
            fd_str = doc.xref_object(fd_xref, compressed=True)
            fd_str = fd_str.replace('/FontName/', '/FontName/' + prefix)
            doc.update_object(fd_xref, fd_str)
    doc.update_object(new_xref, font_str)