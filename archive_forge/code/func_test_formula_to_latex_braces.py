import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, latex', [('Fe(CN)6-3', 'Fe(CN)_{6}^{3-}'), ('((H2O)2OH)12', '((H_{2}O)_{2}OH)_{12}'), ('Fe[CN]6-3', 'Fe[CN]_{6}^{3-}'), ('[(H2O)2OH]12', '[(H_{2}O)_{2}OH]_{12}'), ('Fe{CN}6-3', 'Fe\\{CN\\}_{6}^{3-}'), ('{(H2O)2OH}12', '\\{(H_{2}O)_{2}OH\\}_{12}')])
@requires(parsing_library)
def test_formula_to_latex_braces(species, latex):
    assert formula_to_latex(species) == latex