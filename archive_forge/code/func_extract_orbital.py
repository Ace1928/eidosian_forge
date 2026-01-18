import re
import numpy as np
from xml.dom import minidom
def extract_orbital(orb_xml):
    """
    extract the orbital
    """
    orb = {}
    orb['l'] = str2int(orb_xml.attributes['l'].value)[0]
    orb['n'] = str2int(orb_xml.attributes['n'].value)[0]
    orb['z'] = str2int(orb_xml.attributes['z'].value)[0]
    orb['ispol'] = str2int(orb_xml.attributes['ispol'].value)[0]
    orb['population'] = str2float(orb_xml.attributes['population'].value)[0]
    return orb