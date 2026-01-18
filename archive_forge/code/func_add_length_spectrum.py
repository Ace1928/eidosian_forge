import sys
import tkinter
from tkinter import ttk
from .gui_utilities import UniformDictController, ScrollableFrame
from .geodesics import geodesic_index_to_color, LengthSpectrumError
from ..drilling.exceptions import WordAppearsToBeParabolic
from ..SnapPy import word_as_list # type: ignore
def add_length_spectrum(self, *args, **kwargs):
    self.status_label.configure(text=_default_status_msg, foreground='')
    try:
        if not self.raytracing_view.geodesics.add_length_spectrum(float(self.length_box.get())):
            self.status_label.configure(text='No new geodesics found.', foreground='')
    except LengthSpectrumError as e:
        self.status_label.configure(text=' '.join(e.args), foreground='red')
        return
    except Exception as e:
        self.status_label.configure(text='An error has occurred. See terminal for details.', foreground='red')
    self.raytracing_view.resize_geodesic_params()
    self.populate_geodesics_frame()