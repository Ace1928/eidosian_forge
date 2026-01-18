import time
from .gui import *
from .CyOpenGL import *
from .export_stl import stl
from . import filedialog
from plink.ipython_tools import IPythonTkRoot
def export_stl(self):
    model = self.model_var.get()
    file = filedialog.asksaveasfile(parent=self.parent, title='Save %s model as STL file' % model, defaultextension='.stl', filetypes=[('STL files', '*.stl'), ('All files', '')])
    if file:
        n = 0
        for line in stl(self.polyhedron.facedicts, model=model.lower()):
            file.write(line)
            if n > 100:
                self.root.update_idletasks()
                n = 0
        file.close()