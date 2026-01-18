import os
import sys
import re
import time
from collections.abc import Mapping  # Python 3.5 or newer
from IPython.core.displayhook import DisplayHook
from tkinter.messagebox import askyesno
from .gui import *
from . import filedialog
from .exceptions import SnapPeaFatalError
from .app_menus import HelpMenu, EditMenu, WindowMenu, ListedWindow
from .app_menus import dirichlet_menus, horoball_menus, inside_view_menus, plink_menus
from .app_menus import add_menu, scut, open_html_docs
from .browser import Browser
from .horoviewer import HoroballViewer
from .infowindow import about_snappy, InfoWindow
from .polyviewer import PolyhedronViewer
from .raytracing.inside_viewer import InsideViewer
from .settings import Settings, SettingsDialog
from .phone_home import update_needed
from .SnapPy import SnapPea_interrupt, msg_stream
from .shell import SnapPyInteractiveShellEmbed
from .tkterminal import TkTerm, snappy_path
from plink import LinkEditor
from plink.smooth import Smoother
import site
import pydoc
def open_link_file(self, event=None):
    openfile = filedialog.askopenfile(title='Load Link Projection File', defaultextension='.lnk', filetypes=[('Link and text files', '*.lnk *.txt', 'TEXT'), ('All text files', '', 'TEXT'), ('All files', '')])
    if openfile:
        if not re.search('%\\s*([vV]irtual)*\\s*[lL]ink\\s*[Pp]rojection', openfile.readline()):
            tkMessageBox.showwarning('Bad file', 'This is not a SnapPea link projection file')
            openfile.close()
        else:
            name = openfile.name
            openfile.close()
            line = 'Manifold()\n'
            self.write(line)
            self.interact_handle_input(line)
            self.interact_prompt()
            M = self.IP.user_ns['_']
            M.LE.load(file_name=name)