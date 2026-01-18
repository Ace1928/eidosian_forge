import re
import sys
from optparse import OptionParser
from rdkit import Chem
def build_track_dictionary(smi, stars):
    isotope_track = {}
    if stars == 2:
        matchObj = re.search('\\[\\*\\:([123])\\].*\\[\\*\\:([123])\\]', smi)
        if matchObj:
            isotope_track[matchObj.group(1)] = '1'
            isotope_track[matchObj.group(2)] = '2'
    elif stars == 3:
        matchObj = re.search('\\[\\*\\:([123])\\].*\\[\\*\\:([123])\\].*\\[\\*\\:([123])\\]', smi)
        if matchObj:
            isotope_track[matchObj.group(1)] = '1'
            isotope_track[matchObj.group(2)] = '2'
            isotope_track[matchObj.group(3)] = '3'
    return isotope_track