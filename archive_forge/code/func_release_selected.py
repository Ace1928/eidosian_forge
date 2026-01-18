from ase.gui.i18n import _
import ase.gui.ui as ui
def release_selected(self):
    self.gui.images.set_dynamic(self.gui.images.selected, True)
    self.gui.draw()