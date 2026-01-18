import re
import numpy as np
from xml.dom import minidom
def extract_projector(pro_xml):
    """
    extract the projector
    """
    pro = {}
    pro['l'] = str2int(pro_xml.attributes['l'].value)[0]
    pro['n'] = str2int(pro_xml.attributes['n'].value)[0]
    pro['ref_energy'] = str2float(pro_xml.attributes['ref_energy'].value)[0]
    return pro