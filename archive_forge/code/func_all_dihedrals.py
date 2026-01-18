from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
@property
def all_dihedrals(self):
    """All dihedrals

        Returns a list with indices of atoms in dihedrals for each neighborlist in this instance.
        Atom i forms a dihedral to the atoms inside the tuples in result[i]:
        i -- result[i][x][0] -- result[i][x][1] -- result[i][x][2]
        where x is in range(number of dihedrals from i). See also :data:`unique_dihedrals`.

        **No setter or deleter, only getter**
        """
    if not 'allDihedrals' in self._cache:
        self._cache['allDihedrals'] = []
        distList = self._get_all_x(3)
        for imI in range(len(distList)):
            self._cache['allDihedrals'].append([])
            for iAtom, thirdNeighs in enumerate(distList[imI]):
                self._cache['allDihedrals'][-1].append([])
                if len(thirdNeighs) == 0:
                    continue
                anglesI = self.all_angles[imI][iAtom]
                for lAtom in thirdNeighs:
                    secondNeighs = [angle[-1] for angle in anglesI]
                    firstNeighs = [angle[0] for angle in anglesI]
                    relevantSecondNeighs = [idx for idx in secondNeighs if lAtom in self.all_bonds[imI][idx]]
                    relevantFirstNeighs = [firstNeighs[secondNeighs.index(idx)] for idx in relevantSecondNeighs]
                    for jAtom, kAtom in zip(relevantFirstNeighs, relevantSecondNeighs):
                        tupl = (jAtom, kAtom, lAtom)
                        if len(set((iAtom,) + tupl)) != 4:
                            continue
                        elif tupl in self._cache['allDihedrals'][-1][-1]:
                            continue
                        elif iAtom in tupl:
                            raise RuntimeError('Something is wrong in analysis.all_dihedrals!')
                        self._cache['allDihedrals'][-1][-1].append((jAtom, kAtom, lAtom))
    return self._cache['allDihedrals']