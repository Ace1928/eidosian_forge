import sys
from itertools import combinations
from rdkit import Chem, DataStructs
from rdkit.Chem import rdqueries
def generate_fraggle_fragmentation(mol, verbose=False):
    """ Create all possible fragmentations for molecule
    >>> q = Chem.MolFromSmiles('COc1cc(CN2CCC(NC(=O)c3cncc(C)c3)CC2)c(OC)c2ccccc12')
    >>> fragments = generate_fraggle_fragmentation(q)
    >>> fragments = sorted(['.'.join(sorted(s.split('.'))) for s in fragments])
    >>> fragments
     ['*C(=O)NC1CCN(Cc2cc(OC)c3ccccc3c2OC)CC1',
      '*C(=O)c1cncc(C)c1.*C1CCN(Cc2cc(OC)c3ccccc3c2OC)CC1',
      '*C(=O)c1cncc(C)c1.*Cc1cc(OC)c2ccccc2c1OC',
      '*C(=O)c1cncc(C)c1.*c1cc(OC)c2ccccc2c1OC',
      '*C1CCN(Cc2cc(OC)c3ccccc3c2OC)CC1',
      '*C1CCN(Cc2cc(OC)c3ccccc3c2OC)CC1.*c1cncc(C)c1',
      '*Cc1cc(OC)c2ccccc2c1OC.*NC(=O)c1cncc(C)c1',
      '*Cc1cc(OC)c2ccccc2c1OC.*c1cncc(C)c1',
      '*N1CCC(NC(=O)c2cncc(C)c2)CC1.*c1cc(OC)c2ccccc2c1OC',
      '*NC(=O)c1cncc(C)c1.*c1cc(OC)c2ccccc2c1OC',
      '*NC1CCN(Cc2cc(OC)c3ccccc3c2OC)CC1',
      '*NC1CCN(Cc2cc(OC)c3ccccc3c2OC)CC1.*c1cncc(C)c1',
      '*c1c(CN2CCC(NC(=O)c3cncc(C)c3)CC2)cc(OC)c2ccccc12',
      '*c1c(OC)cc(CN2CCC(NC(=O)c3cncc(C)c3)CC2)c(OC)c1*',
      '*c1cc(CN2CCC(NC(=O)c3cncc(C)c3)CC2)c(OC)c2ccccc12',
      '*c1cc(OC)c2ccccc2c1OC.*c1cncc(C)c1']
  """
    hac = mol.GetNumAtoms()
    acyclic_matching_atoms = mol.GetSubstructMatches(ACYC_SMARTS)
    cyclic_matching_atoms = mol.GetSubstructMatches(CYC_SMARTS)
    if verbose:
        print('Matching Atoms:')
        print('acyclic matching atoms: ', acyclic_matching_atoms)
        print('cyclic matching atoms: ', cyclic_matching_atoms)
    out_fragments = set()
    for bond1, bond2 in combinations(acyclic_matching_atoms, 2):
        fragment = delete_bonds(mol, [bond1, bond2], FTYPE_ACYCLIC, hac)
        if fragment is not None:
            out_fragments.add(fragment)
    for bond1, bond2 in combinations(cyclic_matching_atoms, 2):
        fragment = delete_bonds(mol, [bond1, bond2], FTYPE_CYCLIC, hac)
        if fragment is None:
            continue
        out_fragments.add(fragment)
        for abond in acyclic_matching_atoms:
            fragment = delete_bonds(mol, [bond1, bond2, abond], FTYPE_CYCLIC_ACYCLIC, hac)
            if fragment is not None:
                out_fragments.add(fragment)
    return sorted(out_fragments)