from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def print_to_file(self, filename=None):
    """
        Print the contents of this ``CanvasFrame`` to a postscript
        file.  If no filename is given, then prompt the user for one.

        :param filename: The name of the file to print the tree to.
        :type filename: str
        :rtype: None
        """
    if filename is None:
        ftypes = [('Postscript files', '.ps'), ('All files', '*')]
        filename = asksaveasfilename(filetypes=ftypes, defaultextension='.ps')
        if not filename:
            return
    x0, y0, w, h = self.scrollregion()
    postscript = self._canvas.postscript(x=x0, y=y0, width=w + 2, height=h + 2, pagewidth=w + 2, pageheight=h + 2, pagex=0, pagey=0)
    postscript = postscript.replace(' 0 scalefont ', ' 9 scalefont ')
    with open(filename, 'wb') as f:
        f.write(postscript.encode('utf8'))