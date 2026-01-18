from __future__ import annotations
import matplotlib.pyplot as plt
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.io.vasp import Chgcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.plotting import pretty_plot
def get_dos_plot(args):
    """Plot DOS.

    Args:
        args (dict): Args from argparse.
    """
    vasp_run = Vasprun(args.dos_file)
    dos = vasp_run.complete_dos
    all_dos = {}
    all_dos['Total'] = dos
    structure = vasp_run.final_structure
    if args.site:
        for idx, site in enumerate(structure):
            all_dos[f'Site {idx} {site.specie.symbol}'] = dos.get_site_dos(site)
    if args.element:
        syms = [tok.strip() for tok in args.element[0].split(',')]
        all_dos = {}
        for el, el_dos in dos.get_element_dos().items():
            if el.symbol in syms:
                all_dos[el] = el_dos
    if args.orbital:
        all_dos = dos.get_spd_dos()
    plotter = DosPlotter()
    plotter.add_dos_dict(all_dos)
    return plotter.get_plot()