import pickle
import subprocess
import sys
import weakref
from functools import partial
from ase.gui.i18n import _
from time import time
import numpy as np
from ase import Atoms, __version__
import ase.gui.ui as ui
from ase.gui.defaults import read_defaults
from ase.gui.images import Images
from ase.gui.nanoparticle import SetupNanoparticle
from ase.gui.nanotube import SetupNanotube
from ase.gui.save import save_dialog
from ase.gui.settings import Settings
from ase.gui.status import Status
from ase.gui.surfaceslab import SetupSurfaceSlab
from ase.gui.view import View
def repeat_poll(self, callback, ms, ensure_update=True):
    """Invoke callback(gui=self) every ms milliseconds.

        This is useful for polling a resource for updates to load them
        into the GUI.  The GUI display will be hence be updated after
        each call; pass ensure_update=False to circumvent this.

        Polling stops if the callback function raises StopIteration.

        Example to run a movie manually, then quit::

            from ase.collections import g2
            from ase.gui.gui import GUI

            names = iter(g2.names)

            def main(gui):
                try:
                    name = next(names)
                except StopIteration:
                    gui.window.win.quit()
                else:
                    atoms = g2[name]
                    gui.images.initialize([atoms])

            gui = GUI()
            gui.repeat_poll(main, 30)
            gui.run()"""

    def callbackwrapper():
        try:
            callback(gui=self)
        except StopIteration:
            pass
        finally:
            self.window.win.after(ms, callbackwrapper)
        if ensure_update:
            self.set_frame()
            self.draw()
    self.window.win.after(ms, callbackwrapper)