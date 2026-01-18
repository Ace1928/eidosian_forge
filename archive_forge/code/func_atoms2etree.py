import numpy as np
import xml.etree.ElementTree as ET
from ase.atoms import Atoms
from ase.units import Bohr
from ase.utils import writer
from xml.dom import minidom
def atoms2etree(images):
    """This function creates the XML DOM corresponding
     to the structure for use in write and calculator

    Parameters
    ----------

    images : Atom Object or List of Atoms objects

    Returns
    -------
    root : etree object
        Element tree of exciting input file containing the structure
    """
    if not isinstance(images, (list, tuple)):
        images = [images]
    root = ET.Element('input')
    root.set('{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation', 'http://xml.exciting-code.org/excitinginput.xsd')
    title = ET.SubElement(root, 'title')
    title.text = ''
    structure = ET.SubElement(root, 'structure')
    crystal = ET.SubElement(structure, 'crystal')
    atoms = images[0]
    for vec in atoms.cell:
        basevect = ET.SubElement(crystal, 'basevect')
        basevect.text = '%.14f %.14f %.14f' % tuple(vec / Bohr)
    oldsymbol = ''
    oldrmt = -1
    newrmt = -1
    scaled = atoms.get_scaled_positions()
    for aindex, symbol in enumerate(atoms.get_chemical_symbols()):
        if 'rmt' in atoms.arrays:
            newrmt = atoms.get_array('rmt')[aindex] / Bohr
        if symbol != oldsymbol or newrmt != oldrmt:
            speciesnode = ET.SubElement(structure, 'species', speciesfile='%s.xml' % symbol, chemicalSymbol=symbol)
            oldsymbol = symbol
            if 'rmt' in atoms.arrays:
                oldrmt = atoms.get_array('rmt')[aindex] / Bohr
                if oldrmt > 0:
                    speciesnode.attrib['rmt'] = '%.4f' % oldrmt
        atom = ET.SubElement(speciesnode, 'atom', coord='%.14f %.14f %.14f' % tuple(scaled[aindex]))
        if 'momenta' in atoms.arrays:
            atom.attrib['bfcmt'] = '%.14f %.14f %.14f' % tuple(atoms.get_array('mommenta')[aindex])
    return root