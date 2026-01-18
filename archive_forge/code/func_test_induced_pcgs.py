from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup, DihedralGroup
from sympy.matrices import Matrix
def test_induced_pcgs():
    G = [SymmetricGroup(9).sylow_subgroup(3), SymmetricGroup(20).sylow_subgroup(2), AlternatingGroup(4), DihedralGroup(4), DihedralGroup(10), DihedralGroup(9), SymmetricGroup(3), SymmetricGroup(4)]
    for g in G:
        PcGroup = g.polycyclic_group()
        collector = PcGroup.collector
        gens = list(g.generators)
        ipcgs = collector.induced_pcgs(gens)
        m = []
        for i in ipcgs:
            m.append(collector.exponent_vector(i))
        assert Matrix(m).is_upper