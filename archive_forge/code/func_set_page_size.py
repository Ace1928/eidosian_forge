from reportlab.lib import pagesizes
from reportlab.lib import colors
from reportlab.graphics.shapes import Polygon
from math import pi, sin, cos
from itertools import islice
def set_page_size(self, pagesize, orientation):
    """Set page size of the drawing..

        Arguments:
         - pagesize      Size of the output image, a tuple of pixels (width,
           height, or a string in the reportlab.lib.pagesizes
           set of ISO sizes.
         - orientation   String: 'landscape' or 'portrait'

        """
    if isinstance(pagesize, str):
        pagesize = page_sizes(pagesize)
    elif isinstance(pagesize, tuple):
        pass
    else:
        raise ValueError(f'Page size {pagesize} not recognised')
    shortside, longside = (min(pagesize), max(pagesize))
    orientation = orientation.lower()
    if orientation not in ('landscape', 'portrait'):
        raise ValueError(f'Orientation {orientation} not recognised')
    if orientation == 'landscape':
        self.pagesize = (longside, shortside)
    else:
        self.pagesize = (shortside, longside)