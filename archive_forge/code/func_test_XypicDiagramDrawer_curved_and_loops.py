from sympy.categories.diagram_drawing import _GrowableGrid, ArrowStringDescription
from sympy.categories import (DiagramGrid, Object, NamedMorphism,
from sympy.sets.sets import FiniteSet
def test_XypicDiagramDrawer_curved_and_loops():
    A = Object('A')
    B = Object('B')
    C = Object('C')
    D = Object('D')
    f = NamedMorphism(A, B, 'f')
    g = NamedMorphism(B, C, 'g')
    h = NamedMorphism(D, A, 'h')
    k = NamedMorphism(D, B, 'k')
    d = Diagram([f, g, h, k])
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[r]_{f} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_3mm/[ll]_{h} \\\\\n& C & \n}\n'
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[d]^{f} & \\\\\nB \\ar[r]^{g} & C \\\\\nD \\ar[u]_{k} \\ar@/^3mm/[uu]^{h} & \n}\n'
    assert drawer.draw(d, grid, diagram_format='@+1cm@dr') == '\\xymatrix@+1cm@dr{\nA \\ar[d]^{f} & \\\\\nB \\ar[r]^{g} & C \\\\\nD \\ar[u]_{k} \\ar@/^3mm/[uu]^{h} & \n}\n'
    h1 = NamedMorphism(D, A, 'h1')
    h2 = NamedMorphism(A, D, 'h2')
    k = NamedMorphism(D, B, 'k')
    d = Diagram([f, g, h, k, h1, h2])
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[r]_{f} \\ar@/^3mm/[rr]^{h_{2}} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_7mm/[ll]_{h} \\ar@/_11mm/[ll]_{h_{1}} \\\\\n& C & \n}\n'
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[d]^{f} \\ar@/_3mm/[dd]_{h_{2}} & \\\\\nB \\ar[r]^{g} & C \\\\\nD \\ar[u]_{k} \\ar@/^7mm/[uu]^{h} \\ar@/^11mm/[uu]^{h_{1}} & \n}\n'
    l_A = NamedMorphism(A, A, 'l_A')
    l_D = NamedMorphism(D, D, 'l_D')
    l_C = NamedMorphism(C, C, 'l_C')
    d = Diagram([f, g, h, k, h1, h2, l_A, l_D, l_C])
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[r]_{f} \\ar@/^3mm/[rr]^{h_{2}} \\ar@(u,l)[]^{l_{A}} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_7mm/[ll]_{h} \\ar@/_11mm/[ll]_{h_{1}} \\ar@(r,u)[]^{l_{D}} \\\\\n& C \\ar@(l,d)[]^{l_{C}} & \n}\n'
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[d]^{f} \\ar@/_3mm/[dd]_{h_{2}} \\ar@(r,u)[]^{l_{A}} & \\\\\nB \\ar[r]^{g} & C \\ar@(r,u)[]^{l_{C}} \\\\\nD \\ar[u]_{k} \\ar@/^7mm/[uu]^{h} \\ar@/^11mm/[uu]^{h_{1}} \\ar@(l,d)[]^{l_{D}} & \n}\n'
    l_A_ = NamedMorphism(A, A, 'n_A')
    l_D_ = NamedMorphism(D, D, 'n_D')
    l_C_ = NamedMorphism(C, C, 'n_C')
    d = Diagram([f, g, h, k, h1, h2, l_A, l_D, l_C, l_A_, l_D_, l_C_])
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[r]_{f} \\ar@/^3mm/[rr]^{h_{2}} \\ar@(u,l)[]^{l_{A}} \\ar@/^3mm/@(l,d)[]^{n_{A}} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_7mm/[ll]_{h} \\ar@/_11mm/[ll]_{h_{1}} \\ar@(r,u)[]^{l_{D}} \\ar@/^3mm/@(d,r)[]^{n_{D}} \\\\\n& C \\ar@(l,d)[]^{l_{C}} \\ar@/^3mm/@(d,r)[]^{n_{C}} & \n}\n'
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[d]^{f} \\ar@/_3mm/[dd]_{h_{2}} \\ar@(r,u)[]^{l_{A}} \\ar@/^3mm/@(u,l)[]^{n_{A}} & \\\\\nB \\ar[r]^{g} & C \\ar@(r,u)[]^{l_{C}} \\ar@/^3mm/@(d,r)[]^{n_{C}} \\\\\nD \\ar[u]_{k} \\ar@/^7mm/[uu]^{h} \\ar@/^11mm/[uu]^{h_{1}} \\ar@(l,d)[]^{l_{D}} \\ar@/^3mm/@(d,r)[]^{n_{D}} & \n}\n'