import re
import sys
from rdkit import Chem
from rdkit.Chem import rdMMPA
def fragment_mol(smi, cid):
    mol = Chem.MolFromSmiles(smi)
    outlines = set()
    if mol is None:
        sys.stderr.write("Can't generate mol for: %s\n" % smi)
    else:
        frags = rdMMPA.FragmentMol(mol, pattern='[#6+0;!$(*=,#[!#6])]!@!=!#[*]', resultsAsMols=False)
        for core, chains in frags:
            output = '%s,%s,%s,%s' % (smi, cid, core, chains)
            if not output in outlines:
                outlines.add(output)
        if not outlines:
            outlines.add('%s,%s,,' % (smi, cid))
    return outlines