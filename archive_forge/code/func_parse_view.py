from __future__ import annotations
import argparse
import itertools
from tabulate import tabulate, tabulate_formats
from pymatgen.cli.pmg_analyze import analyze
from pymatgen.cli.pmg_config import configure_pmg
from pymatgen.cli.pmg_plot import plot
from pymatgen.cli.pmg_potcar import generate_potcar
from pymatgen.cli.pmg_structure import analyze_structures
from pymatgen.core import SETTINGS
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Incar, Potcar
def parse_view(args):
    """Handle view commands.

    Args:
        args: Args from command.
    """
    from pymatgen.vis.structure_vtk import StructureVis
    excluded_bonding_elements = args.exclude_bonding[0].split(',') if args.exclude_bonding else []
    struct = Structure.from_file(args.filename[0])
    vis = StructureVis(excluded_bonding_elements=excluded_bonding_elements)
    vis.set_structure(struct)
    vis.show()
    return 0