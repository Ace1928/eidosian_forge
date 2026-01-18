import math
import xmllib
from rdkit.sping.pid import Font
from sping.PDF import PDFCanvas
def allTagCombos(canvas, x, y, font=None, color=None, angle=0):
    """Try out all tags and various combinations of them.
        Starts at given x,y and returns possible next (x,y)."""
    oldDefault = canvas.defaultFont
    if font:
        canvas.defaultFont = font
    oldx = x
    dx = stringWidth(canvas, ' ')
    dy = canvas.defaultFont.size * 1.5
    drawString(canvas, '<b>bold</b>', x, y, color=color, angle=angle)
    x = x + stringWidth(canvas, '<b>bold</b>') + dx
    drawString(canvas, '<i>italic</i>', x, y, color=color, angle=angle)
    x = x + stringWidth(canvas, '<i>italic</i>') + dx
    drawString(canvas, '<u>underline</u>', x, y, color=color, angle=angle)
    x = x + stringWidth(canvas, '<u>underline</u>') + dx
    drawString(canvas, '<super>super</super>', x, y, color=color, angle=angle)
    x = x + stringWidth(canvas, '<super>super</super>') + dx
    drawString(canvas, '<sub>sub</sub>', x, y, color=color, angle=angle)
    y = y + dy
    drawString(canvas, '<b><u>bold+underline</u></b>', oldx, y, color=color, angle=angle)
    x = oldx + stringWidth(canvas, '<b><u>bold+underline</u></b>') + dx
    drawString(canvas, '<super><i>super+italic</i></super>', x, y, color=color, angle=angle)
    x = x + stringWidth(canvas, '<super><i>super+italic</i></super>') + dx
    drawString(canvas, '<b><sub>bold+sub</sub></b>', x, y, color=color, angle=angle)
    y = y + dy
    canvas.defaultFont = oldDefault
    return (oldx, y)