import io
import math
import os
import typing
import weakref
def set_toc_item(doc: fitz.Document, idx: int, dest_dict: OptDict=None, kind: OptInt=None, pno: OptInt=None, uri: OptStr=None, title: OptStr=None, to: point_like=None, filename: OptStr=None, zoom: float=0) -> None:
    """Update TOC item by index.

    It allows changing the item's title and link destination.

    Args:
        idx: (int) desired index of the TOC list, as created by get_toc.
        dest_dict: (dict) destination dictionary as created by get_toc(False).
            Outrules all other parameters. If None, the remaining parameters
            are used to make a dest dictionary.
        kind: (int) kind of link (fitz.LINK_GOTO, etc.). If None, then only the
            title will be updated. If fitz.LINK_NONE, the TOC item will be deleted.
        pno: (int) page number (1-based like in get_toc). Required if fitz.LINK_GOTO.
        uri: (str) the URL, required if fitz.LINK_URI.
        title: (str) the new title. No change if None.
        to: (point-like) destination on the target page. If omitted, (72, 36)
            will be used as taget coordinates.
        filename: (str) destination filename, required for fitz.LINK_GOTOR and
            fitz.LINK_LAUNCH.
        name: (str) a destination name for fitz.LINK_NAMED.
        zoom: (float) a zoom factor for the target location (fitz.LINK_GOTO).
    """
    xref = doc.get_outline_xrefs()[idx]
    page_xref = 0
    if type(dest_dict) is dict:
        if dest_dict['kind'] == fitz.LINK_GOTO:
            pno = dest_dict['page']
            page_xref = doc.page_xref(pno)
            page_height = doc.page_cropbox(pno).height
            to = dest_dict.get('to', fitz.Point(72, 36))
            to.y = page_height - to.y
            dest_dict['to'] = to
        action = getDestStr(page_xref, dest_dict)
        if not action.startswith('/A'):
            raise ValueError('bad bookmark dest')
        color = dest_dict.get('color')
        if color:
            color = list(map(float, color))
            if len(color) != 3 or min(color) < 0 or max(color) > 1:
                raise ValueError('bad color value')
        bold = dest_dict.get('bold', False)
        italic = dest_dict.get('italic', False)
        flags = italic + 2 * bold
        collapse = dest_dict.get('collapse')
        return doc._update_toc_item(xref, action=action[2:], title=title, color=color, flags=flags, collapse=collapse)
    if kind == fitz.LINK_NONE:
        return doc.del_toc_item(idx)
    if kind is None and title is None:
        return None
    if kind is None:
        return doc._update_toc_item(xref, action=None, title=title)
    if kind == fitz.LINK_GOTO:
        if pno is None or pno not in range(1, doc.page_count + 1):
            raise ValueError('bad page number')
        page_xref = doc.page_xref(pno - 1)
        page_height = doc.page_cropbox(pno - 1).height
        if to is None:
            to = fitz.Point(72, page_height - 36)
        else:
            to = fitz.Point(to)
            to.y = page_height - to.y
    ddict = {'kind': kind, 'to': to, 'uri': uri, 'page': pno, 'file': filename, 'zoom': zoom}
    action = getDestStr(page_xref, ddict)
    if action == '' or not action.startswith('/A'):
        raise ValueError('bad bookmark dest')
    return doc._update_toc_item(xref, action=action[2:], title=title)