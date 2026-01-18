import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ase.io.xsd import SetChild, _write_xsd_html
from ase import Atoms
def read_arcfile(fd, index):
    images = []
    fd.readline()
    pbc = 'ON' in fd.readline()
    L = fd.readline()
    while L != '':
        if '!' not in L:
            L = fd.readline()
            continue
        if pbc:
            L = fd.readline()
            cell = [float(d) for d in L.split()[1:]]
        else:
            fd.readline()
        symbols = []
        coords = []
        while True:
            line = fd.readline()
            L = line.split()
            if not line or 'end' in L:
                break
            symbols.append(L[0])
            coords.append([float(x) for x in L[1:4]])
        if pbc:
            image = Atoms(symbols, positions=coords, cell=cell, pbc=pbc)
        else:
            image = Atoms(symbols, positions=coords, pbc=pbc)
        images.append(image)
        L = fd.readline()
    if not index:
        return images
    else:
        return images[index]