from sympy.categories.diagram_drawing import _GrowableGrid, ArrowStringDescription
from sympy.categories import (DiagramGrid, Object, NamedMorphism,
from sympy.sets.sets import FiniteSet
def test_XypicDiagramDrawer_line():
    A = Object('A')
    B = Object('B')
    C = Object('C')
    D = Object('D')
    E = Object('E')
    f = NamedMorphism(A, B, 'f')
    g = NamedMorphism(B, C, 'g')
    h = NamedMorphism(C, D, 'h')
    i = NamedMorphism(D, E, 'i')
    d = Diagram([f, g, h, i])
    grid = DiagramGrid(d, layout='sequential')
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[r]^{f} & B \\ar[r]^{g} & C \\ar[r]^{h} & D \\ar[r]^{i} & E \n}\n'
    grid = DiagramGrid(d, layout='sequential', transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[d]^{f} \\\\\nB \\ar[d]^{g} \\\\\nC \\ar[d]^{h} \\\\\nD \\ar[d]^{i} \\\\\nE \n}\n'