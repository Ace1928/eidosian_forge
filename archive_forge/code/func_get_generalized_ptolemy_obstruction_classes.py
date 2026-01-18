from . import matrix
from . import homology
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVariety import PtolemyVariety
from .utilities import MethodMappingList
def get_generalized_ptolemy_obstruction_classes(manifold, N):
    """
    See SnapPy.pyx for documentation

    >>> from snappy import Manifold
    >>> M = Manifold("4_1")
    >>> get_generalized_ptolemy_obstruction_classes(M, 2)
    [PtolemyGeneralizedObstructionClass([0, 0, 0, 0]), PtolemyGeneralizedObstructionClass([1, 0, 0, 1])]
    >>> get_generalized_ptolemy_obstruction_classes(M, 3)
    [PtolemyGeneralizedObstructionClass([0, 0, 0, 0]), PtolemyGeneralizedObstructionClass([2, 0, 0, 1])]
    >>> get_generalized_ptolemy_obstruction_classes(M, 4)
    [PtolemyGeneralizedObstructionClass([0, 0, 0, 0]), PtolemyGeneralizedObstructionClass([3, 0, 0, 1]), PtolemyGeneralizedObstructionClass([2, 0, 0, 2])]
    >>> get_generalized_ptolemy_obstruction_classes(M, 5)
    [PtolemyGeneralizedObstructionClass([0, 0, 0, 0]), PtolemyGeneralizedObstructionClass([4, 0, 0, 1])]

    >>> M = Manifold("m202")
    >>> len(get_generalized_ptolemy_obstruction_classes(M, 2))
    4
    >>> len(get_generalized_ptolemy_obstruction_classes(M, 3))
    5
    >>> len(get_generalized_ptolemy_obstruction_classes(M, 4))
    10

    >>> M = Manifold("m207")
    >>> len(get_generalized_ptolemy_obstruction_classes(M, 2))
    2
    >>> len(get_generalized_ptolemy_obstruction_classes(M, 3))
    14
    >>> len(get_generalized_ptolemy_obstruction_classes(M, 4))
    3
    """
    H2_elements, explain_columns = get_obstruction_classes(manifold, N)
    filtered_H2_elements = []
    units = [x for x in range(N) if _gcd(x, N) == 1]
    already_seen = set()
    for H2_element in H2_elements:
        if not tuple(H2_element) in already_seen:
            filtered_H2_elements.append(H2_element)
            for u in units:
                already_seen.add(tuple([x * u % N for x in H2_element]))
    return [PtolemyGeneralizedObstructionClass(H2_element, index=index, N=N, manifold=manifold) for index, H2_element in enumerate(filtered_H2_elements)]