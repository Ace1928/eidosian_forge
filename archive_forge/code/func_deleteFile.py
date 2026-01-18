import xml.dom.minidom, xml.sax.saxutils
import os, time, fcntl
from xdg.Exceptions import ParsingError
def deleteFile(self, item):
    """Remove a recently used file, by URI, from the list.
        """
    if item in self.RecentFiles:
        self.RecentFiles.remove(item)