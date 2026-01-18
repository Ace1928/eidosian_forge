import sys
import os
import webbrowser
from urllib.request import pathname2url
from .gui import *
from . import __file__ as snappy_dir
from .infowindow import about_snappy, InfoWindow
from .version import version
import shutil
def dirichlet_menus(self):
    """
    Menus for the standalone Dirichlet viewer.  Called by the view Frame, not the
    parent Toplevel.
    """
    self.menubar = menubar = Tk_.Menu(self.parent)
    if sys.platform == 'darwin':
        Python_menu = Tk_.Menu(menubar, name='apple')
        Python_menu.add_command(label='About SnapPy...', command=self.main_window.about_window)
        menubar.add_cascade(label='SnapPy', menu=Python_menu)
    File_menu = Tk_.Menu(menubar, name='file')
    add_menu(self.master, File_menu, 'Open...', None, 'disabled')
    add_menu(self.master, File_menu, 'Save as...', None, 'disabled')
    File_menu.add_command(label='Save Image...', command=self.master.save_image)
    Export_menu = Tk_.Menu(File_menu, name='export')
    File_menu.add_cascade(label='Export as STL...', menu=Export_menu)
    Export_menu.add_command(label='Export STL', command=self.export_stl)
    Export_menu.add_command(label='Export Cutout STL', command=self.export_cutout_stl)
    File_menu.add_separator()
    add_menu(self.master, File_menu, 'Close', command=self.master.close)
    menubar.add_cascade(label='File', menu=File_menu)
    menubar.add_cascade(label='Edit ', menu=EditMenu(menubar, self.master.edit_actions))
    menubar.add_cascade(label='Window', menu=WindowMenu(menubar))
    help_menu = HelpMenu(menubar)
    help_menu.extra_command(label=help_polyhedron_viewer_label, command=self.help_window)
    help_menu.activate([help_polyhedron_viewer_label, help_report_bugs_label])
    self.menubar.add_cascade(label='Help', menu=help_menu)