from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def selectFieldRelative(canvas, title, value, options, xR, yR, width, height):
    """same as textFieldAbsolute except the x and y are relative to the canvas coordinate transform"""
    xA, yA = canvas.absolutePosition(xR, yR)
    return selectFieldAbsolute(canvas, title, value, options, xA, yA, width, height)