from ase import Atoms
from ase.io import read
from ase.io.jsonio import read_json
from ase.dft.kpoints import BandPath
from ase.cli.main import CLIError
from ase.io.formats import UnknownFileTypeError
def plot_reciprocal_cell(path, output=None):
    import matplotlib.pyplot as plt
    path.plot()
    if output:
        plt.savefig(output)
    else:
        plt.show()