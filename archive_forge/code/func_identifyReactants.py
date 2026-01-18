import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def identifyReactants(reaction, output=False):
    rxn = AllChem.ChemicalReaction(reaction)
    AllChem.RemoveMappingNumbersFromReactions(rxn)
    if output:
        print('--- identifyReactants ---')
    reactants = rxn.GetReactants()
    products = rxn.GetProducts()
    uniqueReactants, reactantSmiles = utils.uniqueMolecules(reactants)
    uniqueProducts, productSmiles = utils.uniqueMolecules(products)
    unmodifiedReactants, unmodifiedProducts = _detectObviousReagents(reactantSmiles, productSmiles)
    if output:
        print('  >>> Found reagents in reactants:', unmodifiedReactants)
        print('  >>> Found reagents in products:', unmodifiedProducts)
    if len(products) == len(unmodifiedProducts):
        unmodifiedProducts = set()
    uniquePotentialReactants = [r for r in sorted(set(uniqueReactants.values()))]
    uniquePotentialProducts = [p for p in sorted(set(uniqueProducts.values())) if p not in unmodifiedProducts]
    rfps = [MoleculeDetails(reactants[r]) for r in uniquePotentialReactants]
    pfps = [MoleculeDetails(products[p]) for p in uniquePotentialProducts]
    rfpsPrep = [(MoleculeDetails(reactants[r]), reactantSmiles[r] in frequentReagents) for r in uniquePotentialReactants]
    reacts, unmappedProdAtoms = _getBestCombination(rfpsPrep, pfps, output=output)
    if np.array(reacts).shape == (1, 0):
        rfpsPrep = [(MoleculeDetails(reactants[r]), 0) for r in uniquePotentialReactants]
        reacts, unmappedProdAtoms = _getBestCombination(rfpsPrep, pfps, output=output)
    reacts = _findMissingReactiveReactants(rfps, pfps, reacts, unmappedProdAtoms, output=output)
    finalreacts = []
    for i in reacts:
        temp = [uniquePotentialReactants[j] for j in i]
        finalreacts.append(set(temp))
    return (finalreacts, unmodifiedReactants, unmodifiedProducts)