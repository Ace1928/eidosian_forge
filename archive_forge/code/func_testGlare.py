import itertools
import math
import operator
import random
from functools import reduce
def testGlare():
    a = RGroups(makeFakeSidechains('aldehydes', 1000))
    b = RGroups(makeFakeSidechains('boronic_acids', 1500))
    lib = Library([a, b])
    props = [Property('mw', 0, 0, 500, 230.1419), Property('alogp', 1, -2.4, 5, 2.212749), Property('tpsa', 2, 0, 90, 24.5)]
    glare = Glare()
    glare.optimize(lib, props)
    for reactant_idx, rgroup in enumerate(lib.rgroups):
        print(f'Reactants for reactant {reactant_idx}')
        for reactant in rgroup.sidechains:
            print(reactant.name)