from ase import Atoms
from ase.io import read
from ase.io.jsonio import read_json
from ase.dft.kpoints import BandPath
from ase.cli.main import CLIError
from ase.io.formats import UnknownFileTypeError
def obj2bandpath(obj):
    if isinstance(obj, BandPath):
        print('Object is a band path')
        print(obj)
        return obj
    if isinstance(getattr(obj, 'path', None), BandPath):
        print(f'Object contains a bandpath: {obj}')
        path = obj.path
        print(path)
        return path
    if isinstance(obj, Atoms):
        print(f'Atoms object: {obj}')
        print('Determining standard form of Bravais lattice:')
        lat = obj.cell.get_bravais_lattice(pbc=obj.pbc)
        print(lat.description())
        print('Showing default bandpath')
        return lat.bandpath(density=0)
    raise CLIError(f'Strange object: {obj}')