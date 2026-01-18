import io
import math
import os
import typing
import weakref
def set_old_widths(xref, widths, dwidths):
    """Restore the old '/W' and '/DW' in subsetted font.

        If either parameter is None or evaluates to False, the corresponding
        dictionary key will be set to null.
        """
    df = doc.xref_get_key(xref, 'DescendantFonts')
    if df[0] != 'array':
        return None
    df_xref = int(df[1][1:-1].replace('0 R', ''))
    if (type(widths) is not str or not widths) and doc.xref_get_key(df_xref, 'W')[0] != 'null':
        doc.xref_set_key(df_xref, 'W', 'null')
    else:
        doc.xref_set_key(df_xref, 'W', widths)
    if (type(dwidths) is not str or not dwidths) and doc.xref_get_key(df_xref, 'DW')[0] != 'null':
        doc.xref_set_key(df_xref, 'DW', 'null')
    else:
        doc.xref_set_key(df_xref, 'DW', dwidths)
    return None