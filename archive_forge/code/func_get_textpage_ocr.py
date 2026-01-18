import io
import math
import os
import typing
import weakref
def get_textpage_ocr(page: fitz.Page, flags: int=0, language: str='eng', dpi: int=72, full: bool=False, tessdata: str=None) -> fitz.TextPage:
    """Create a Textpage from combined results of normal and OCR text parsing.

    Args:
        flags: (int) control content becoming part of the result.
        language: (str) specify expected language(s). Deafault is "eng" (English).
        dpi: (int) resolution in dpi, default 72.
        full: (bool) whether to OCR the full page image, or only its images (default)
    """
    fitz.CheckParent(page)
    if not TESSDATA_PREFIX and (not tessdata):
        raise RuntimeError('No OCR support: TESSDATA_PREFIX not set')

    def full_ocr(page, dpi, language, flags):
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        ocr_pdf = fitz.Document('pdf', pix.pdfocr_tobytes(compress=False, language=language, tessdata=tessdata))
        ocr_page = ocr_pdf.load_page(0)
        unzoom = page.rect.width / ocr_page.rect.width
        ctm = fitz.Matrix(unzoom, unzoom) * page.derotation_matrix
        tpage = ocr_page.get_textpage(flags=flags, matrix=ctm)
        ocr_pdf.close()
        pix = None
        tpage.parent = weakref.proxy(page)
        return tpage
    if full is True:
        return full_ocr(page, dpi, language, flags)
    tpage = page.get_textpage(flags=flags)
    for block in page.get_text('dict', flags=fitz.TEXT_PRESERVE_IMAGES)['blocks']:
        if block['type'] != 1:
            continue
        bbox = fitz.Rect(block['bbox'])
        if bbox.width <= 3 or bbox.height <= 3:
            continue
        exception_types = (RuntimeError, mupdf.FzErrorBase)
        if fitz.mupdf_version_tuple < (1, 24):
            exception_types = RuntimeError
        try:
            pix = fitz.Pixmap(block['image'])
            if pix.n - pix.alpha != 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.alpha:
                pix = fitz.Pixmap(pix, 0)
            imgdoc = fitz.Document('pdf', pix.pdfocr_tobytes(language=language, tessdata=tessdata))
            imgpage = imgdoc.load_page(0)
            pix = None
            imgrect = imgpage.rect
            shrink = fitz.Matrix(1 / imgrect.width, 1 / imgrect.height)
            mat = shrink * block['transform']
            imgpage.extend_textpage(tpage, flags=0, matrix=mat)
            imgdoc.close()
        except exception_types:
            if g_exceptions_verbose:
                fitz.exception_info()
            tpage = None
            fitz.message('Falling back to full page OCR')
            return full_ocr(page, dpi, language, flags)
    return tpage