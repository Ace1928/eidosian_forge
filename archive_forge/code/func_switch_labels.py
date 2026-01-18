import re
import sys
from optparse import OptionParser
from rdkit import Chem
def switch_labels(track, stars, smi):
    if stars > 1:
        if track['1'] != '1':
            smi = re.sub('\\[\\*\\:1\\]', '[*:XX' + track['1'] + 'XX]', smi)
        if track['2'] != '2':
            smi = re.sub('\\[\\*\\:2\\]', '[*:XX' + track['2'] + 'XX]', smi)
        if stars == 3:
            if track['3'] != '3':
                smi = re.sub('\\[\\*\\:3\\]', '[*:XX' + track['3'] + 'XX]', smi)
        smi = re.sub('XX', '', smi)
    return smi