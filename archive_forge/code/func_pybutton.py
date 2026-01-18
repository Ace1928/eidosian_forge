from ase.gui.i18n import _
import ase.data
import ase.gui.ui as ui
from ase import Atoms
def pybutton(title, callback):
    """A button for displaying Python code.

    When pressed, it opens a window displaying some Python code, or an error
    message if no Python code is ready.
    """
    return ui.Button('Python', pywindow, title, callback)