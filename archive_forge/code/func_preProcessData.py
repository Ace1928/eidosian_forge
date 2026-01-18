from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import shapes
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Drawing
def preProcessData(self, data):
    """preprocess and return a new array with at least one row
        and column (use a None) if needed, and all rows the same
        length (adding Nones if needed)

        """
    if not data:
        return [[None]]
    max_row = max([len(x) for x in data])
    for rowNo, row in enumerate(data):
        if len(row) < max_row:
            row.extend([None] * (max_row - len(row)))
    return data