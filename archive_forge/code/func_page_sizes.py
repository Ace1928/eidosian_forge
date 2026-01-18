from reportlab.lib import pagesizes
from reportlab.lib import colors
from reportlab.graphics.shapes import Polygon
from math import pi, sin, cos
from itertools import islice
def page_sizes(size):
    """Convert size string into a Reportlab pagesize.

    Arguments:
     - size - A string representing a standard page size, eg 'A4' or 'LETTER'

    """
    sizes = {'A0': pagesizes.A0, 'A1': pagesizes.A1, 'A2': pagesizes.A2, 'A3': pagesizes.A3, 'A4': pagesizes.A4, 'A5': pagesizes.A5, 'A6': pagesizes.A6, 'B0': pagesizes.B0, 'B1': pagesizes.B1, 'B2': pagesizes.B2, 'B3': pagesizes.B3, 'B4': pagesizes.B4, 'B5': pagesizes.B5, 'B6': pagesizes.B6, 'ELEVENSEVENTEEN': pagesizes.ELEVENSEVENTEEN, 'LEGAL': pagesizes.LEGAL, 'LETTER': pagesizes.LETTER}
    try:
        return sizes[size]
    except KeyError:
        raise ValueError(f'{size} not in list of page sizes') from None