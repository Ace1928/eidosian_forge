from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires('numpy')
def test_ReactionSystem__html_tables():
    r1 = Reaction({'A': 2}, {'A'}, name='R1')
    r2 = Reaction({'A'}, {'A': 2}, name='R2')
    rs = ReactionSystem([r1, r2])
    ut, unc = rs.unimolecular_html_table()
    assert unc == {0}
    from chempy.printing import html
    assert html(ut, with_name=False) == u'<table><tr><td>A</td><td ><a title="1: A → 2 A">R2</a></td></tr></table>'
    bt, bnc = rs.bimolecular_html_table()
    assert bnc == {1}
    assert html(bt, with_name=False) == u'<table><th></th><th>A</th>\n<tr><td>A</td><td ><a title="0: 2 A → A">R1</a></td></tr></table>'