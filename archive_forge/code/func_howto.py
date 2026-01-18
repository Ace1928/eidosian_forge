import os, time, webbrowser
from .gui import *
from . import smooth
from .vertex import Vertex
from .arrow import Arrow
from .crossings import Crossing, ECrossing
from .colors import Palette
from .dialog import InfoDialog
from .manager import LinkManager
from .viewer import LinkViewer
from .version import version
from .ipython_tools import IPythonTkRoot
def howto(self):
    doc_file = os.path.join(os.path.dirname(__file__), 'doc', 'index.html')
    doc_path = os.path.abspath(doc_file)
    url = 'file:' + pathname2url(doc_path)
    try:
        webbrowser.open(url)
    except:
        tkMessageBox.showwarning('Not found!', 'Could not open URL\n(%s)' % url)