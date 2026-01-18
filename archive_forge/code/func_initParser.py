import argparse
import sys
from rdkit import Chem, Geometry
from rdkit.Chem import rdDepictor
def initParser():
    """ Initialize the parser """
    parser = argparse.ArgumentParser(description='Create aligned depiction')
    parser.add_argument('--pattern', '-p', metavar='SMARTS', default=None, dest='patt')
    parser.add_argument('--smiles', default=False, action='store_true', dest='useSmiles', help='Set if core and input are SMILES strings')
    parser.add_argument('-o', dest='outF', type=argparse.FileType('w'), default=sys.stdout, metavar='OUTFILE', help='Specify a file to take the output. If missing, uses stdout.')
    parser.add_argument('core', metavar='core')
    parser.add_argument('mol', metavar='molecule', help='')
    return parser