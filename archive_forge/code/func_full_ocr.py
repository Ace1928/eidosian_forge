import io
import math
import os
import typing
import weakref
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