from ase.gui.i18n import _
import ase.gui.ui as ui
def show_selected(self):
    self.gui.images.visible[self.gui.images.selected] = True
    self.gui.draw()