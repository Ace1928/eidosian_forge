import os
import numpy as np
import xml.etree.ElementTree as ET
from ase.io.exciting import atoms2etree
from ase.units import Bohr, Hartree
from ase.calculators.calculator import PropertyNotImplementedError
from xml.dom import minidom
def dicttoxml(self, pdict, element):
    for key, value in pdict.items():
        if isinstance(value, str) and key == 'text()':
            element.text = value
        elif isinstance(value, str):
            element.attrib[key] = value
        elif isinstance(value, list):
            for item in value:
                self.dicttoxml(item, ET.SubElement(element, key))
        elif isinstance(value, dict):
            if element.findall(key) == []:
                self.dicttoxml(value, ET.SubElement(element, key))
            else:
                self.dicttoxml(value, element.findall(key)[0])
        else:
            print('cannot deal with', key, '=', value)