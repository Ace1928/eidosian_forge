import sys
import tkinter
from tkinter import ttk
from .gui_utilities import UniformDictController, ScrollableFrame
from .geodesics import geodesic_index_to_color, LengthSpectrumError
from ..drilling.exceptions import WordAppearsToBeParabolic
from ..SnapPy import word_as_list # type: ignore
def add_word(self, *args, **kwargs):
    word = self.word_entry.get()
    if len(word) == 0:
        self.status_label.configure(text='Word is empty', foreground='red')
        return
    try:
        n = self.raytracing_view.geodesics.get_mcomplex().num_generators
        word_as_list(word, n)
    except ValueError:
        self.status_label.configure(text=word + ' contains non-generators', foreground='red')
        return
    try:
        index = self.raytracing_view.geodesics.add_word(word)
    except WordAppearsToBeParabolic:
        self.status_label.configure(text=word + ' is parabolic', foreground='red')
        return
    self.status_label.configure(text=_default_status_msg, foreground='')
    self.raytracing_view.resize_geodesic_params()
    self.raytracing_view.enable_geodesic(index)
    if self.raytracing_view.disable_edges_for_geodesics():
        self.inside_viewer.update_edge_and_insphere_controllers()
    self.raytracing_view.update_geodesic_data_and_redraw()
    self.populate_geodesics_frame()