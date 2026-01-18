from IPython.display import display, Javascript, Latex, SVG, HTML, Markdown
from IPython.core.magic import  (
from IPython.core import magic_arguments
@cell_magic
def javascript(self, line, cell):
    """Run the cell block of Javascript code

        Starting with IPython 8.0 %%javascript is pending deprecation to be replaced
        by a more flexible system

        Please See https://github.com/ipython/ipython/issues/13376
        """
    display(Javascript(cell))