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
def edit_settings(self):
    terminal.can_quit = False
    if sys.platform == 'darwin':
        self.window.deletecommand('::tk::mac::ShowPreferences')
    else:
        apple_menu = self.menubar.children['apple']
        apple_menu.entryconfig(2, state='disabled')
    dialog = SettingsDialog(self.window, self.settings)
    terminal.add_blocker(dialog, 'Changes to your settings will be lost if you quit SnapPy now.')
    dialog.run()
    terminal.remove_blocker(dialog)
    if dialog.okay:
        answer = askyesno('Save?', 'Do you want to save these settings?')
        if answer:
            self.settings.write_settings()
    if sys.platform == 'darwin':
        self.window.createcommand('::tk::mac::ShowPreferences', self.edit_settings)
    else:
        apple_menu.entryconfig(2, state='active')
    self.can_quit = True