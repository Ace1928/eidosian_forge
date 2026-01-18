from __future__ import print_function, unicode_literals
import six
def extract_widths(font_face):
    widths = dict(iter_charwidths(font_face))
    widths.update(bibtex_widths)
    return widths