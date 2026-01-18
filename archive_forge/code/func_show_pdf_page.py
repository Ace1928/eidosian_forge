import io
import math
import os
import typing
import weakref
def show_pdf_page(page, rect, src, pno=0, keep_proportion=True, overlay=True, oc=0, rotate=0, clip=None) -> int:
    """Show page number 'pno' of PDF 'src' in rectangle 'rect'.

    Args:
        rect: (rect-like) where to place the source image
        src: (document) source PDF
        pno: (int) source page number
        keep_proportion: (bool) do not change width-height-ratio
        overlay: (bool) put in foreground
        oc: (xref) make visibility dependent on this OCG / OCMD (which must be defined in the target PDF)
        rotate: (int) degrees (multiple of 90)
        clip: (rect-like) part of source page rectangle
    Returns:
        xref of inserted object (for reuse)
    """

    def calc_matrix(sr, tr, keep=True, rotate=0):
        """Calculate transformation matrix from source to target rect.

        Notes:
            The product of four matrices in this sequence: (1) translate correct
            source corner to origin, (2) rotate, (3) scale, (4) translate to
            target's top-left corner.
        Args:
            sr: source rect in PDF (!) coordinate system
            tr: target rect in PDF coordinate system
            keep: whether to keep source ratio of width to height
            rotate: rotation angle in degrees
        Returns:
            Transformation matrix.
        """
        smp = (sr.tl + sr.br) / 2.0
        tmp = (tr.tl + tr.br) / 2.0
        m = fitz.Matrix(1, 0, 0, 1, -smp.x, -smp.y) * fitz.Matrix(rotate)
        sr1 = sr * m
        fw = tr.width / sr1.width
        fh = tr.height / sr1.height
        if keep:
            fw = fh = min(fw, fh)
        m *= fitz.Matrix(fw, fh)
        m *= fitz.Matrix(1, 0, 0, 1, tmp.x, tmp.y)
        return fitz.JM_TUPLE(m)
    fitz.CheckParent(page)
    doc = page.parent
    if not doc.is_pdf or not src.is_pdf:
        raise ValueError('is no PDF')
    if rect.is_empty or rect.is_infinite:
        raise ValueError('rect must be finite and not empty')
    while pno < 0:
        pno += src.page_count
    src_page = src[pno]
    if src_page.get_contents() == []:
        raise ValueError('nothing to show - source page empty')
    tar_rect = rect * ~page.transformation_matrix
    src_rect = src_page.rect if not clip else src_page.rect & clip
    if src_rect.is_empty or src_rect.is_infinite:
        raise ValueError('clip must be finite and not empty')
    src_rect = src_rect * ~src_page.transformation_matrix
    matrix = calc_matrix(src_rect, tar_rect, keep=keep_proportion, rotate=rotate)
    ilst = [i[1] for i in doc.get_page_xobjects(page.number)]
    ilst += [i[7] for i in doc.get_page_images(page.number)]
    ilst += [i[4] for i in doc.get_page_fonts(page.number)]
    n = 'fzFrm'
    i = 0
    _imgname = n + '0'
    while _imgname in ilst:
        i += 1
        _imgname = n + str(i)
    isrc = src._graft_id
    if doc._graft_id == isrc:
        raise ValueError('source document must not equal target')
    gmap = doc.Graftmaps.get(isrc, None)
    if gmap is None:
        gmap = fitz.Graftmap(doc)
        doc.Graftmaps[isrc] = gmap
    pno_id = (isrc, pno)
    xref = doc.ShownPages.get(pno_id, 0)
    if overlay:
        page.wrap_contents()
    xref = page._show_pdf_page(src_page, overlay=overlay, matrix=matrix, xref=xref, oc=oc, clip=src_rect, graftmap=gmap, _imgname=_imgname)
    doc.ShownPages[pno_id] = xref
    return xref