import os
import warnings
from collections import namedtuple
from importlib.util import find_spec
from io import BytesIO
import numpy
from rdkit import Chem
from rdkit import RDConfig
from rdkit import rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit.Chem.Draw.MolDrawing import MolDrawing
from rdkit.Chem.Draw.rdMolDraw2D import *
def calcAtomGaussians(mol, a=0.03, step=0.02, weights=None):
    """
useful things to do with these:
fig.axes[0].imshow(z,cmap=cm.gray,interpolation='bilinear',origin='lower',extent=(0,1,0,1))
fig.axes[0].contour(x,y,z,20,colors='k')

fig=Draw.MolToMPL(m);
contribs=Crippen.rdMolDescriptors._CalcCrippenContribs(m)
logps,mrs=zip(*contribs)
x,y,z=Draw.calcAtomGaussians(m,0.03,step=0.01,weights=logps)
fig.axes[0].imshow(z,cmap=cm.jet,interpolation='bilinear',origin='lower',extent=(0,1,0,1))
fig.axes[0].contour(x,y,z,20,colors='k',alpha=0.5)
fig.savefig('coumlogps.colored.png',bbox_inches='tight')


  """
    x = numpy.arange(0, 1, step)
    y = numpy.arange(0, 1, step)
    X, Y = numpy.meshgrid(x, y)
    if weights is None:
        weights = [1.0] * mol.GetNumAtoms()
    Z = _bivariate_normal(X, Y, a, a, mol._atomPs[0][0], mol._atomPs[0][1]) * weights[0]
    for i in range(1, mol.GetNumAtoms()):
        Zp = _bivariate_normal(X, Y, a, a, mol._atomPs[i][0], mol._atomPs[i][1])
        Z += Zp * weights[i]
    return (X, Y, Z)