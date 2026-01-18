from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
@property
def all_angles(self):
    """All angles

        A list with indices of atoms in angles for each neighborlist in *self*.
        Atom i forms an angle to the atoms inside the tuples in result[i]:
        i -- result[i][x][0] -- result[i][x][1]
        where x is in range(number of angles from i). See also :data:`unique_angles`.

        **No setter or deleter, only getter**
        """
    if not 'allAngles' in self._cache:
        self._cache['allAngles'] = []
        distList = self._get_all_x(2)
        for imI in range(len(distList)):
            self._cache['allAngles'].append([])
            for iAtom, secNeighs in enumerate(distList[imI]):
                self._cache['allAngles'][-1].append([])
                if len(secNeighs) == 0:
                    continue
                firstNeighs = self.all_bonds[imI][iAtom]
                for kAtom in secNeighs:
                    relevantFirstNeighs = [idx for idx in firstNeighs if kAtom in self.all_bonds[imI][idx]]
                    for jAtom in relevantFirstNeighs:
                        self._cache['allAngles'][-1][-1].append((jAtom, kAtom))
    return self._cache['allAngles']