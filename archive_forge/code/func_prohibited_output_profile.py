from __future__ import (absolute_import, division, print_function)
from stringprep import (
from unicodedata import normalize
from ansible.module_utils.six import text_type
def prohibited_output_profile(string):
    """RFC4013 Prohibited output profile implementation."""
    if is_ral_string(string):
        is_prohibited_bidi_ch = in_table_d2
        bidi_table = 'D.2'
    else:
        is_prohibited_bidi_ch = in_table_d1
        bidi_table = 'D.1'
    RFC = 'RFC4013'
    for c in string:
        if in_table_c12(c):
            raise ValueError('%s: prohibited non-ASCII space characters that cannot be replaced (C.1.2).' % RFC)
        if in_table_c21_c22(c):
            raise ValueError('%s: prohibited control characters (C.2.1).' % RFC)
        if in_table_c3(c):
            raise ValueError('%s: prohibited private Use characters (C.3).' % RFC)
        if in_table_c4(c):
            raise ValueError('%s: prohibited non-character code points (C.4).' % RFC)
        if in_table_c5(c):
            raise ValueError('%s: prohibited surrogate code points (C.5).' % RFC)
        if in_table_c6(c):
            raise ValueError('%s: prohibited inappropriate for plain text characters (C.6).' % RFC)
        if in_table_c7(c):
            raise ValueError('%s: prohibited inappropriate for canonical representation characters (C.7).' % RFC)
        if in_table_c8(c):
            raise ValueError('%s: prohibited change display properties / deprecated characters (C.8).' % RFC)
        if in_table_c9(c):
            raise ValueError('%s: prohibited tagging characters (C.9).' % RFC)
        if is_prohibited_bidi_ch(c):
            raise ValueError('%s: prohibited bidi characters (%s).' % (RFC, bidi_table))
        if in_table_a1(c):
            raise ValueError('%s: prohibited unassigned code points (A.1).' % RFC)