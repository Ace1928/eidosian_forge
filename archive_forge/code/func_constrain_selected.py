from ase.gui.i18n import _
import ase.gui.ui as ui
def constrain_selected(self):
    self.gui.images.set_dynamic(self.gui.images.selected, False)
    self.gui.draw()