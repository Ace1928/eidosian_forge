from reportlab.lib import colors
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.markers import makeEmptySquare, makeFilledSquare
from reportlab.graphics.charts.markers import makeFilledDiamond, makeSmiley
from reportlab.graphics.charts.markers import makeFilledCircle, makeEmptyCircle
from Bio.Graphics import _write
def draw_to_file(self, output_file, title):
    """Write the comparative plot to a file.

        Arguments:
         - output_file - The name of the file to output the information to,
           or a handle to write to.
         - title - A title to display on the graphic.

        """
    width, height = self.page_size
    cur_drawing = Drawing(width, height)
    self._draw_title(cur_drawing, title, width, height)
    start_x = inch * 0.5
    end_x = width - inch * 0.5
    end_y = height - 1.5 * inch
    start_y = 0.5 * inch
    self._draw_scatter_plot(cur_drawing, start_x, start_y, end_x, end_y)
    return _write(cur_drawing, output_file, self.output_format)