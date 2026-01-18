import io
import math
import os
import typing
import weakref
def set_toc(doc: fitz.Document, toc: list, collapse: int=1) -> int:
    """Create new outline tree (table of contents, TOC).

    Args:
        toc: (list, tuple) each entry must contain level, title, page and
            optionally top margin on the page. None or '()' remove the TOC.
        collapse: (int) collapses entries beyond this level. Zero or None
            shows all entries unfolded.
    Returns:
        the number of inserted items, or the number of removed items respectively.
    """
    if doc.is_closed or doc.is_encrypted:
        raise ValueError('document closed or encrypted')
    if not doc.is_pdf:
        raise ValueError('is no PDF')
    if not toc:
        return len(doc._delToC())
    if type(toc) not in (list, tuple):
        raise ValueError("'toc' must be list or tuple")
    toclen = len(toc)
    page_count = doc.page_count
    t0 = toc[0]
    if type(t0) not in (list, tuple):
        raise ValueError('items must be sequences of 3 or 4 items')
    if t0[0] != 1:
        raise ValueError('hierarchy level of item 0 must be 1')
    for i in list(range(toclen - 1)):
        t1 = toc[i]
        t2 = toc[i + 1]
        if not -1 <= t1[2] <= page_count:
            raise ValueError('row %i: page number out of range' % i)
        if type(t2) not in (list, tuple) or len(t2) not in (3, 4):
            raise ValueError('bad row %i' % (i + 1))
        if type(t2[0]) is not int or t2[0] < 1:
            raise ValueError('bad hierarchy level in row %i' % (i + 1))
        if t2[0] > t1[0] + 1:
            raise ValueError('bad hierarchy level in row %i' % (i + 1))
    old_xrefs = doc._delToC()
    old_xrefs = []
    xref = [0] + old_xrefs
    xref[0] = doc._getOLRootNumber()
    if toclen > len(old_xrefs):
        for i in range(toclen - len(old_xrefs)):
            xref.append(doc.get_new_xref())
    lvltab = {0: 0}
    olitems = [{'count': 0, 'first': -1, 'last': -1, 'xref': xref[0]}]
    for i in range(toclen):
        o = toc[i]
        lvl = o[0]
        title = fitz.get_pdf_str(o[1])
        pno = min(doc.page_count - 1, max(0, o[2] - 1))
        page_xref = doc.page_xref(pno)
        page_height = doc.page_cropbox(pno).height
        top = fitz.Point(72, page_height - 36)
        dest_dict = {'to': top, 'kind': fitz.LINK_GOTO}
        if o[2] < 0:
            dest_dict['kind'] = fitz.LINK_NONE
        if len(o) > 3:
            if type(o[3]) in (int, float):
                dest_dict['to'] = fitz.Point(72, page_height - o[3])
            else:
                dest_dict = o[3].copy() if type(o[3]) is dict else dest_dict
                if 'to' not in dest_dict:
                    dest_dict['to'] = top
                else:
                    page = doc[pno]
                    point = fitz.Point(dest_dict['to'])
                    point.y = page.cropbox.height - point.y
                    point = point * page.rotation_matrix
                    dest_dict['to'] = (point.x, point.y)
        d = {}
        d['first'] = -1
        d['count'] = 0
        d['last'] = -1
        d['prev'] = -1
        d['next'] = -1
        d['dest'] = getDestStr(page_xref, dest_dict)
        d['top'] = dest_dict['to']
        d['title'] = title
        d['parent'] = lvltab[lvl - 1]
        d['xref'] = xref[i + 1]
        d['color'] = dest_dict.get('color')
        d['flags'] = dest_dict.get('italic', 0) + 2 * dest_dict.get('bold', 0)
        lvltab[lvl] = i + 1
        parent = olitems[lvltab[lvl - 1]]
        if dest_dict.get('collapse') or (collapse and lvl > collapse):
            parent['count'] -= 1
        else:
            parent['count'] += 1
        if parent['first'] == -1:
            parent['first'] = i + 1
            parent['last'] = i + 1
        else:
            d['prev'] = parent['last']
            prev = olitems[parent['last']]
            prev['next'] = i + 1
            parent['last'] = i + 1
        olitems.append(d)
    for i, ol in enumerate(olitems):
        txt = '<<'
        if ol['count'] != 0:
            txt += '/Count %i' % ol['count']
        try:
            txt += ol['dest']
        except Exception:
            if g_exceptions_verbose:
                fitz.exception_info()
            pass
        try:
            if ol['first'] > -1:
                txt += '/First %i 0 R' % xref[ol['first']]
        except Exception:
            if g_exceptions_verbose:
                fitz.exception_info()
            pass
        try:
            if ol['last'] > -1:
                txt += '/Last %i 0 R' % xref[ol['last']]
        except Exception:
            if g_exceptions_verbose:
                fitz.exception_info()
            pass
        try:
            if ol['next'] > -1:
                txt += '/Next %i 0 R' % xref[ol['next']]
        except Exception:
            if g_exceptions_verbose:
                fitz.exception_info()
            pass
        try:
            if ol['parent'] > -1:
                txt += '/Parent %i 0 R' % xref[ol['parent']]
        except Exception:
            if g_exceptions_verbose:
                fitz.exception_info()
            pass
        try:
            if ol['prev'] > -1:
                txt += '/Prev %i 0 R' % xref[ol['prev']]
        except Exception:
            if g_exceptions_verbose:
                fitz.exception_info()
            pass
        try:
            txt += '/Title' + ol['title']
        except Exception:
            if g_exceptions_verbose:
                fitz.exception_info()
            pass
        if ol.get('color') and len(ol['color']) == 3:
            txt += '/C[ %g %g %g]' % tuple(ol['color'])
        if ol.get('flags', 0) > 0:
            txt += '/F %i' % ol['flags']
        if i == 0:
            txt += '/Type/Outlines'
        txt += '>>'
        doc.update_object(xref[i], txt)
    doc.init_doc()
    return toclen