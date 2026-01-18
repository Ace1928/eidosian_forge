import argparse
import operator
import sys
from collections import Counter, defaultdict, namedtuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import FilterCatalog, RDConfig, rdMolDescriptors
def buildFilterCatalog():
    inhousefilter = pd.read_csv(f'{RDConfig.RDContribDir}/NIBRSubstructureFilters/SubstructureFilter_HitTriaging_wPubChemExamples.csv')
    inhouseFiltersCat = FilterCatalog.FilterCatalog()
    for i in range(inhousefilter.shape[0]):
        mincount = 1
        if inhousefilter['MIN_COUNT'][i] != 0:
            mincount = int(inhousefilter['MIN_COUNT'][i])
        pname = inhousefilter['PATTERN_NAME'][i]
        sname = inhousefilter['SET_NAME'][i]
        pname_final = '{0}_min({1})__{2}__{3}__{4}'.format(pname, mincount, inhousefilter['SEVERITY_SCORE'][i], inhousefilter['COVALENT'][i], inhousefilter['SPECIAL_MOL'][i])
        fil = FilterCatalog.SmartsMatcher(pname_final, inhousefilter['SMARTS'][i], mincount)
        inhouseFiltersCat.AddEntry(FilterCatalog.FilterCatalogEntry(pname_final, fil))
        inhouseFiltersCat.GetEntry(i).SetProp('Scope', sname)
    return inhouseFiltersCat