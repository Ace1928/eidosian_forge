from ase.gui.i18n import _
import ase.gui.ui as ui
def hide_selected(self):
    self.gui.images.visible[self.gui.images.selected] = False
    self.gui.draw()