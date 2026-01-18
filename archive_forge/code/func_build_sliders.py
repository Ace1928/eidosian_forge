from .gui import *
from .CyOpenGL import (HoroballScene, OpenGLOrthoWidget,
from plink.ipython_tools import IPythonTkRoot
import os
import sys
def build_sliders(self):
    nbhd = self.nbhd
    if nbhd is None:
        return
    self.cusp_vars = []
    self.cusp_colors = []
    self.tie_vars = []
    num_cusps = nbhd.num_cusps()
    if num_cusps > 1:
        self.eye_label.grid(row=0, column=0, sticky=Tk_.E, pady=0)
        self.tie_label.grid(row=0, column=1, sticky=Tk_.E, pady=0)
    else:
        self.eye_label.grid_forget()
        self.tie_label.grid_forget()
    for n in range(num_cusps):
        disp = float(nbhd.stopping_displacement(which_cusp=n))
        nbhd.set_displacement(disp, which_cusp=n)
        if nbhd and nbhd.num_cusps() > 1:
            eye_button = ttk.Radiobutton(self.slider_frame, text='', variable=self.eye_var, takefocus=False, value=n, command=self.set_eye)
            self.eye_buttons.append(eye_button)
            eye_button.grid(row=n + 1, column=0)
            tie_var = Tk_.IntVar(self)
            tie_var.set(nbhd.get_tie(n))
            self.tie_vars.append(tie_var)
            tie_button = ttk.Checkbutton(self.slider_frame, variable=tie_var, takefocus=False, command=self.rebuild)
            tie_button.index = n
            tie_button.grid(row=n + 1, column=1)
            self.tie_buttons.append(tie_button)
        R, G, B, A = GetColor(nbhd.original_index(n))
        self.cusp_colors.append('#%.3x%.3x%.3x' % (int(R * 4095), int(G * 4095), int(B * 4095)))
        self.cusp_vars.append(Tk_.IntVar(self))
        self.slider_frames.append(Tk_.Frame(self.slider_frame, borderwidth=0))
        self.slider_frames[n].grid(row=n + 1, column=2, sticky=Tk_.EW, padx=6, pady=1)
        slider = Tk_.Scale(self.slider_frames[n], showvalue=0, from_=0, to=100, width=11, length=200, orient=Tk_.HORIZONTAL, background=self.cusp_colors[n], troughcolor=self.bgcolor, borderwidth=1, relief=Tk_.FLAT, variable=Tk_.DoubleVar(self))
        slider.index = n
        slider.stamp = 0
        slider.bind('<ButtonPress-1>', self.start_radius)
        slider.bind('<ButtonRelease-1>', self.end_radius)
        slider.grid(padx=(0, 20), pady=0, sticky=Tk_.W)
        self.cusp_sliders.append(slider)
        volume_label = ttk.Label(self.slider_frame, width=6)
        volume_label.grid(row=n + 1, column=3, sticky=Tk_.W)
        self.volume_labels.append(volume_label)