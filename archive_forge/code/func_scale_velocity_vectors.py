from ase.gui.i18n import _
import ase.gui.ui as ui
def scale_velocity_vectors(self):
    self.gui.velocity_vector_scale = float(self.velocity_vector_scale.value)
    self.gui.draw()
    return True