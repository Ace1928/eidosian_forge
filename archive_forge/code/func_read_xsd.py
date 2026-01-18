import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ase import Atoms
from ase.utils import writer
def read_xsd(fd):
    tree = ET.parse(fd)
    root = tree.getroot()
    atomtreeroot = root.find('AtomisticTreeRoot')
    if atomtreeroot.find('SymmetrySystem') is not None:
        symmetrysystem = atomtreeroot.find('SymmetrySystem')
        mappingset = symmetrysystem.find('MappingSet')
        mappingfamily = mappingset.find('MappingFamily')
        system = mappingfamily.find('IdentityMapping')
        coords = list()
        cell = list()
        formula = str()
        for atom in system:
            if atom.tag == 'Atom3d':
                symbol = atom.get('Components')
                formula += symbol
                xyz = atom.get('XYZ')
                if xyz:
                    coord = [float(coord) for coord in xyz.split(',')]
                else:
                    coord = [0.0, 0.0, 0.0]
                coords.append(coord)
            elif atom.tag == 'SpaceGroup':
                avec = [float(vec) for vec in atom.get('AVector').split(',')]
                bvec = [float(vec) for vec in atom.get('BVector').split(',')]
                cvec = [float(vec) for vec in atom.get('CVector').split(',')]
                cell.append(avec)
                cell.append(bvec)
                cell.append(cvec)
        atoms = Atoms(formula, cell=cell, pbc=True)
        atoms.set_scaled_positions(coords)
        return atoms
    elif atomtreeroot.find('Molecule') is not None:
        system = atomtreeroot.find('Molecule')
        coords = list()
        formula = str()
        for atom in system:
            if atom.tag == 'Atom3d':
                symbol = atom.get('Components')
                formula += symbol
                xyz = atom.get('XYZ')
                coord = [float(coord) for coord in xyz.split(',')]
                coords.append(coord)
        atoms = Atoms(formula, pbc=False)
        atoms.set_scaled_positions(coords)
        return atoms