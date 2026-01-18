import numpy as np
from ase.build.general_surface import surface
from ase.geometry import get_layers
from ase.symbols import string2symbols
def surfaces_with_termination(lattice, indices, layers, vacuum=None, tol=1e-10, termination=None, return_all=False, verbose=False):
    """Create surface from a given lattice and Miller indices with a given
        termination

        Parameters
        ==========
        lattice: Atoms object or str
            Bulk lattice structure of alloy or pure metal.  Note that the
            unit-cell must be the conventional cell - not the primitive cell.
            One can also give the chemical symbol as a string, in which case the
            correct bulk lattice will be generated automatically.
        indices: sequence of three int
            Surface normal in Miller indices (h,k,l).
        layers: int
            Number of equivalent layers of the slab. (not the same as the layers
            you choose from for terminations)
        vacuum: float
            Amount of vacuum added on both sides of the slab.
        termination: str
            the atoms you wish to be in the top layer. There may be many such
            terminations, this function returns all terminations with the same
            atomic composition.
            e.g. 'O' will return oxygen terminated surfaces.
            e.g.'TiO' will return surfaces terminated with layers containing both O
            and Ti
        Returns:
        return_surfs: List
            a list of surfaces that match the specifications given

    """
    lats = translate_lattice(lattice, indices)
    return_surfs = []
    check = []
    check2 = []
    for item in lats:
        too_similar = False
        surf = surface(item, indices, layers, vacuum=vacuum, tol=tol)
        surf.wrap(pbc=[True] * 3)
        positions = surf.get_scaled_positions().flatten()
        for i, value in enumerate(positions):
            if value >= 1 - tol:
                positions[i] -= 1
        surf.set_scaled_positions(np.reshape(positions, (len(surf), 3)))
        z_layers, hs = get_layers(surf, (0, 0, 1))
        top_layer = [i for i, val in enumerate(z_layers == max(z_layers)) if val]
        if termination is not None:
            comp = [surf.get_chemical_symbols()[a] for a in top_layer]
            term = string2symbols(termination)
            check = [a for a in comp if a not in term]
            check2 = [a for a in term if a not in comp]
        if len(return_surfs) > 0:
            pos_diff = [a.get_positions() - surf.get_positions() for a in return_surfs]
            for i, su in enumerate(pos_diff):
                similarity_test = su.flatten() < tol * 1000
                if similarity_test.all():
                    too_similar = True
        if too_similar:
            continue
        if return_all is True:
            pass
        elif check != [] or check2 != []:
            continue
        return_surfs.append(surf)
    return return_surfs