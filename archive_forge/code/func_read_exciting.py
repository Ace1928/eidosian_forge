import numpy as np
import xml.etree.ElementTree as ET
from ase.atoms import Atoms
from ase.units import Bohr
from ase.utils import writer
from xml.dom import minidom
def read_exciting(fileobj, index=-1):
    """Reads structure from exiting xml file.

    Parameters
    ----------
    fileobj: file object
        File handle from which data should be read.

    Other parameters
    ----------------
    index: integer -1
        Not used in this implementation.
    """
    doc = ET.parse(fileobj)
    root = doc.getroot()
    speciesnodes = root.find('structure').iter('species')
    symbols = []
    positions = []
    basevects = []
    atoms = None
    for speciesnode in speciesnodes:
        symbol = speciesnode.get('speciesfile').split('.')[0]
        natoms = speciesnode.iter('atom')
        for atom in natoms:
            x, y, z = atom.get('coord').split()
            positions.append([float(x), float(y), float(z)])
            symbols.append(symbol)
    if 'scale' in doc.find('structure/crystal').attrib:
        scale = float(str(doc.find('structure/crystal').attrib['scale']))
    else:
        scale = 1
    if 'stretch' in doc.find('structure/crystal').attrib:
        a, b, c = doc.find('structure/crystal').attrib['stretch'].text.split()
        stretch = np.array([float(a), float(b), float(c)])
    else:
        stretch = np.array([1.0, 1.0, 1.0])
    basevectsn = root.findall('structure/crystal/basevect')
    for basevect in basevectsn:
        x, y, z = basevect.text.split()
        basevects.append(np.array([float(x) * Bohr * stretch[0], float(y) * Bohr * stretch[1], float(z) * Bohr * stretch[2]]) * scale)
    atoms = Atoms(symbols=symbols, cell=basevects)
    atoms.set_scaled_positions(positions)
    if 'molecule' in root.find('structure').attrib.keys():
        if root.find('structure').attrib['molecule']:
            atoms.set_pbc(False)
    else:
        atoms.set_pbc(True)
    return atoms