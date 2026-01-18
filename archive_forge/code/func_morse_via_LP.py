from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def morse_via_LP(link, solver='GLPK'):
    """
    An integer linear program which computes the Morse number of the given
    link diagram.

    EXAMPLES::

        sage: K = RationalTangle(23, 43).denominator_closure()
        sage: morse, details = morse_via_LP(K)
        sage: morse
        2
        sage: morse_via_LP(Link('8_20'))[0]
        3
    """
    LP = MixedIntegerLinearProgram(maximization=False, solver=solver)
    hor_cross = LP.new_variable(binary=True)
    vert_cross = LP.new_variable(binary=True)
    flat_edge = LP.new_variable(binary=True)
    large_edge = LP.new_variable(binary=True)
    exterior = LP.new_variable(binary=True)
    faces = link.faces()
    LP.add_constraint(sum((exterior[i] for i, F in enumerate(faces))) == 1)
    for c in link.crossings:
        LP.add_constraint(hor_cross[c] + vert_cross[c] == 1)
        for ce in c.entry_points():
            s = CrossingStrand(c, ce.strand_index)
            t = s.opposite()
            LP.add_constraint(flat_edge[s] == flat_edge[t])
            LP.add_constraint(flat_edge[s] + large_edge[s] + large_edge[t] == 1)
    for i, face in enumerate(faces):
        eqn = 0
        for cs in face:
            flat = hor_cross if cs.strand_index % 2 == 0 else vert_cross
            eqn += flat[cs.crossing] + flat_edge[cs] + 2 * large_edge[cs]
        LP.add_constraint(eqn == 2 * len(face) - 2 + 4 * exterior[i])
    LP.set_objective(sum(large_edge.values()))
    morse = int(LP.solve())
    assert morse % 2 == 0
    return (morse // 2, LP.get_values([hor_cross, vert_cross, flat_edge, large_edge, exterior]))