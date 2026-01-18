import bisect
import numpy
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors, rdPartialCharges
def pySlogP_VSA_(mol, bins=None, force=1):
    """ *Internal Use Only*
  """
    if not force:
        try:
            res = mol._slogpVSA
        except AttributeError:
            pass
        else:
            if res.all():
                return res
    if bins is None:
        bins = logpBins
    Crippen._Init()
    propContribs = Crippen._GetAtomContribs(mol, force=force)
    volContribs = _LabuteHelper(mol)
    ans = numpy.zeros(len(bins) + 1, 'd')
    for i in range(len(propContribs)):
        prop = propContribs[i]
        vol = volContribs[i + 1]
        if prop is not None:
            bin_ = bisect.bisect_right(bins, prop[0])
            ans[bin_] += vol
    mol._slogpVSA = ans
    return ans