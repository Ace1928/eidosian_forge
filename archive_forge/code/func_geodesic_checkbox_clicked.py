import sys
import tkinter
from tkinter import ttk
from .gui_utilities import UniformDictController, ScrollableFrame
from .geodesics import geodesic_index_to_color, LengthSpectrumError
from ..drilling.exceptions import WordAppearsToBeParabolic
from ..SnapPy import word_as_list # type: ignore
def geodesic_checkbox_clicked(self):
    if self.raytracing_view.disable_edges_for_geodesics():
        self.inside_viewer.update_edge_and_insphere_controllers()
    self.raytracing_view.update_geodesic_data_and_redraw()