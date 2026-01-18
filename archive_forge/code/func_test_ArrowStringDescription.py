from sympy.categories.diagram_drawing import _GrowableGrid, ArrowStringDescription
from sympy.categories import (DiagramGrid, Object, NamedMorphism,
from sympy.sets.sets import FiniteSet
def test_ArrowStringDescription():
    astr = ArrowStringDescription('cm', '', None, '', '', 'd', 'r', '_', 'f')
    assert str(astr) == '\\ar[dr]_{f}'
    astr = ArrowStringDescription('cm', '', 12, '', '', 'd', 'r', '_', 'f')
    assert str(astr) == '\\ar[dr]_{f}'
    astr = ArrowStringDescription('cm', '^', 12, '', '', 'd', 'r', '_', 'f')
    assert str(astr) == '\\ar@/^12cm/[dr]_{f}'
    astr = ArrowStringDescription('cm', '', 12, 'r', '', 'd', 'r', '_', 'f')
    assert str(astr) == '\\ar[dr]_{f}'
    astr = ArrowStringDescription('cm', '', 12, 'r', 'u', 'd', 'r', '_', 'f')
    assert str(astr) == '\\ar@(r,u)[dr]_{f}'
    astr = ArrowStringDescription('cm', '', 12, 'r', 'u', 'd', 'r', '_', 'f')
    assert str(astr) == '\\ar@(r,u)[dr]_{f}'
    astr = ArrowStringDescription('cm', '', 12, 'r', 'u', 'd', 'r', '_', 'f')
    astr.arrow_style = '{-->}'
    assert str(astr) == '\\ar@(r,u)@{-->}[dr]_{f}'
    astr = ArrowStringDescription('cm', '_', 12, '', '', 'd', 'r', '_', 'f')
    astr.arrow_style = '{-->}'
    assert str(astr) == '\\ar@/_12cm/@{-->}[dr]_{f}'