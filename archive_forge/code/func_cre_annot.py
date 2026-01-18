import io
import math
import os
import typing
import weakref
def cre_annot(lnk, xref_dst, pno_src, ctm):
    """Create annotation object string for a passed-in link."""
    r = lnk['from'] * ctm
    rect = '%g %g %g %g' % tuple(r)
    if lnk['kind'] == fitz.LINK_GOTO:
        txt = fitz.annot_skel['goto1']
        idx = pno_src.index(lnk['page'])
        p = lnk['to'] * ctm
        annot = txt % (xref_dst[idx], p.x, p.y, lnk['zoom'], rect)
    elif lnk['kind'] == fitz.LINK_GOTOR:
        if lnk['page'] >= 0:
            txt = fitz.annot_skel['gotor1']
            pnt = lnk.get('to', fitz.Point(0, 0))
            if type(pnt) is not fitz.Point:
                pnt = fitz.Point(0, 0)
            annot = txt % (lnk['page'], pnt.x, pnt.y, lnk['zoom'], lnk['file'], lnk['file'], rect)
        else:
            txt = fitz.annot_skel['gotor2']
            to = fitz.get_pdf_str(lnk['to'])
            to = to[1:-1]
            f = lnk['file']
            annot = txt % (to, f, rect)
    elif lnk['kind'] == fitz.LINK_LAUNCH:
        txt = fitz.annot_skel['launch']
        annot = txt % (lnk['file'], lnk['file'], rect)
    elif lnk['kind'] == fitz.LINK_URI:
        txt = fitz.annot_skel['uri']
        annot = txt % (lnk['uri'], rect)
    else:
        annot = ''
    return annot