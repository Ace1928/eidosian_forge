import bisect
import numpy
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors, rdPartialCharges
def pyPEOE_VSA_(mol, bins=None, force=1):
    """ *Internal Use Only*
  """
    if not force:
        try:
            res = mol._peoeVSA
        except AttributeError:
            pass
        else:
            if res.all():
                return res
    if bins is None:
        bins = chgBins
    Crippen._Init()
    rdPartialCharges.ComputeGasteigerCharges(mol)
    propContribs = []
    for at in mol.GetAtoms():
        p = at.GetProp('_GasteigerCharge')
        try:
            v = float(p)
        except ValueError:
            v = 0.0
        propContribs.append(v)
    volContribs = _LabuteHelper(mol)
    ans = numpy.zeros(len(bins) + 1, 'd')
    for i in range(len(propContribs)):
        prop = propContribs[i]
        vol = volContribs[i + 1]
        if prop is not None:
            bin_ = bisect.bisect_right(bins, prop)
            ans[bin_] += vol
    mol._peoeVSA = ans
    return ans