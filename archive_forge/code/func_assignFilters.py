import argparse
import operator
import sys
from collections import Counter, defaultdict, namedtuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import FilterCatalog, RDConfig, rdMolDescriptors
def assignFilters(data, nameSmilesColumn='smiles'):
    results = []
    inhouseFiltersCat = buildFilterCatalog()
    NO_filter = '[#7,#8]'
    sma = Chem.MolFromSmarts(NO_filter, mergeHs=True)
    for smi in data[nameSmilesColumn]:
        qc, NO_filter, fracNO, co, sc, sm = [np.NaN] * 6
        try:
            mol = Chem.MolFromSmiles(smi)
            numHeavyAtoms = mol.GetNumHeavyAtoms()
            numNO = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7,#8]')))
            fracNO = float(numNO) / numHeavyAtoms
            entries = inhouseFiltersCat.GetMatches(mol)
            if len(list(entries)):
                fs, sev, cov, spm = ([] for _ in range(4))
                for entry in entries:
                    pname = entry.GetDescription()
                    n, s, c, m = pname.split('__')
                    fs.append(entry.GetProp('Scope') + '_' + n)
                    sev.append(int(s))
                    cov.append(int(c))
                    spm.append(int(m))
                qc = ' | '.join(fs)
                if sev.count(2):
                    sc = 10
                else:
                    sc = sum(sev)
                co = sum(cov)
                sm = sum(spm)
            else:
                qc = 'no match'
                sc = 0
                co = 0
                sm = 0
            if not mol.HasSubstructMatch(sma):
                NO_filter = 'no_oxygen_or_nitrogen'
            else:
                NO_filter = 'no match'
        except Exception:
            print('Failed on compound {0}\n'.format(smi))
            pass
        results.append(FilterMatch(qc, NO_filter, fracNO, co, sm, sc))
    return results