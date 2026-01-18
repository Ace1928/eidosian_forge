from ase.gui.i18n import _
import ase.gui.ui as ui
def scale_force_vectors(self):
    self.gui.force_vector_scale = float(self.force_vector_scale.value)
    self.gui.draw()
    return True