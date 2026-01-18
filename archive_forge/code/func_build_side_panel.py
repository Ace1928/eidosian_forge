import sys
import os
from .gui import *
from .polyviewer import PolyhedronViewer
from .horoviewer import HoroballViewer
from .CyOpenGL import GetColor
from .app_menus import browser_menus
from . import app_menus
from .number import Number
from . import database
from .exceptions import SnapPeaFatalError
from plink import LinkViewer, LinkEditor
from plink.ipython_tools import IPythonTkRoot
from spherogram.links.orthogonal import OrthogonalLinkDiagram
def build_side_panel(self):
    side_panel = ttk.Frame(self)
    side_panel.grid_rowconfigure(5, weight=1)
    filling = ttk.Labelframe(side_panel, text='Filling Curves', padding=(10, 10))
    self.filling_canvas = None
    num_cusps = self.manifold.num_cusps()
    if num_cusps > 3:
        filling.grid_propagate(False)
        filling.configure(width=cusp_box_width + 20, height=int(3.5 * cusp_box_height))
        filling.grid_columnconfigure(0, weight=1)
        filling.grid_rowconfigure(0, weight=1)
        filling_scrollbar = ttk.Scrollbar(filling)
        filling_scrollbar.grid(row=0, column=1, sticky=Tk_.NS)
        self.filling_canvas = canvas = Tk_.Canvas(filling, bd=0, highlightthickness=0, yscrollcommand=filling_scrollbar.set, scrollregion=(0, 0, cusp_box_width + 20, cusp_box_height * num_cusps + 10))
        if sys.platform == 'darwin':
            canvas.configure(background=self.style.groupBG)
        canvas.grid(row=0, column=0, sticky=Tk_.NSEW)
        filling_scrollbar.config(command=canvas.yview)
    self.filling_vars = []
    cusp_parent = self.filling_canvas if self.filling_canvas else filling
    for n in range(num_cusps):
        R, G, B, A = GetColor(n)
        color = '#%.3x%.3x%.3x' % (int(R * 4095), int(G * 4095), int(B * 4095))
        cusp = ttk.Labelframe(cusp_parent, text='Cusp %d' % n)
        mer_var = Tk_.StringVar(self, value='0')
        long_var = Tk_.StringVar(self, value='0')
        self.filling_vars.append((mer_var, long_var))
        Tk_.Label(cusp, width=0, background=color, bd=1).grid(row=0, column=0, rowspan=2, sticky=Tk_.NS, padx=4, pady=8)
        ttk.Label(cusp, text='Meridian:').grid(row=0, column=1, sticky=Tk_.E)
        ttk.Label(cusp, text='Longitude:').grid(row=1, column=1, sticky=Tk_.E)
        meridian = Spinbox(cusp, width=4, textvariable=mer_var, from_=-1000, to=1000, increment=1, name=':%s:0' % n, validate='focusout', validatecommand=(self.register(self.validate_coeff), '%P', '%W'))
        meridian.bind('<Return>', self.do_filling)
        meridian.grid(row=0, column=2, sticky=Tk_.W, padx=0, pady=3)
        longitude = Spinbox(cusp, width=4, textvariable=long_var, from_=-1000, to=1000, increment=1, name=':%s:1' % n, validate='focusout', validatecommand=(self.register(self.validate_coeff), '%P', '%W'))
        longitude.bind('<Return>', self.do_filling)
        longitude.grid(row=1, column=2, sticky=Tk_.W, padx=0, pady=3)
        if self.filling_canvas:
            self.filling_canvas.create_window(0, n * cusp_box_height, anchor=Tk_.NW, window=cusp)
        else:
            cusp.grid(row=n, pady=8)
    filling.grid(row=0, column=0, padx=10, pady=10)
    modify = ttk.Labelframe(side_panel, text='Modify', padding=(10, 10))
    self.fill_button = ttk.Button(modify, text='Fill', command=self.do_filling).pack(padx=10, pady=5, expand=True, fill=Tk_.X)
    ttk.Button(modify, text='Retriangulate', command=self.retriangulate).pack(padx=10, pady=5, expand=True, fill=Tk_.X)
    modify.grid(row=1, column=0, padx=10, pady=10, sticky=Tk_.EW)
    create = ttk.Labelframe(side_panel, text='Create', padding=(10, 10))
    ttk.Button(create, text='Drill ...', command=self.drill).pack(padx=10, pady=5, expand=True, fill=Tk_.X)
    ttk.Button(create, text='Cover ...', command=self.cover).pack(padx=10, pady=5, expand=True, fill=Tk_.X)
    create.grid(row=2, column=0, padx=10, pady=10, sticky=Tk_.EW)
    return side_panel