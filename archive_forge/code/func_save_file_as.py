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
def save_file_as(self, event=None):
    savefile = filedialog.asksaveasfile(parent=self.window, mode='w', title='Save Transcript as a Python script', defaultextension='.py', filetypes=[('Python and text files', '*.py *.ipy *.txt', 'TEXT'), ('All text files', '', 'TEXT'), ('All files', '')])
    if savefile:
        savefile.write('#!/usr/bin/env/python\n# This script was saved by SnapPy on %s.\n' % time.asctime())
        inputs = self.IP.history_manager.input_hist_raw
        results = self.IP.history_manager.output_hist
        for n in range(1, len(inputs)):
            savefile.write('\n' + re.sub('\n+', '\n', inputs[n]) + '\n')
            try:
                output = repr(results[n]).split('\n')
            except:
                continue
            for line in output:
                savefile.write('#' + line + '\n')
        savefile.close()