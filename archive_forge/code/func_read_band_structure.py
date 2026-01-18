from ase.io.jsonio import read_json
from ase.spectrum.band_structure import BandStructure
from ase.cli.main import CLIError
def read_band_structure(filename):
    bs = read_json(filename)
    if not isinstance(bs, BandStructure):
        raise CLIError(f'Expected band structure, but file contains: {bs}')
    return bs