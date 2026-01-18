import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, unicode', [('NH4+', u'NH₄⁺'), ('H2O', u'H₂O'), ('C6H6+', u'C₆H₆⁺'), ('Fe(CN)6-3', u'Fe(CN)₆³⁻'), ('C18H38+2', u'C₁₈H₃₈²⁺'), ('((H2O)2OH)12', u'((H₂O)₂OH)₁₂'), ('[(H2O)2OH]12', u'[(H₂O)₂OH]₁₂'), ('{(H2O)2OH}12', u'{(H₂O)₂OH}₁₂'), ('NaCl', u'NaCl'), ('NaCl(s)', u'NaCl(s)'), ('e-(aq)', u'e⁻(aq)'), ('Ca+2(aq)', u'Ca²⁺(aq)'), ('.NO2(g)', u'⋅NO₂(g)'), ('.NH2', u'⋅NH₂'), ('ONOOH', u'ONOOH'), ('.ONOO', u'⋅ONOO'), ('.NO3-2', u'⋅NO₃²⁻'), ('alpha-FeOOH(s)', u'α-FeOOH(s)'), ('epsilon-Zn(OH)2(s)', u'ε-Zn(OH)₂(s)'), ('Na2CO3..7H2O(s)', u'Na₂CO₃·7H₂O(s)'), ('Na2CO3..1H2O(s)', u'Na₂CO₃·H₂O(s)'), ('K4[Fe(CN)6]', 'K₄[Fe(CN)₆]'), ('K4[Fe(CN)6](s)', 'K₄[Fe(CN)₆](s)'), ('K4[Fe(CN)6](aq)', 'K₄[Fe(CN)₆](aq)'), ('[Fe(H2O)6][Fe(CN)6]..19H2O', '[Fe(H₂O)₆][Fe(CN)₆]·19H₂O'), ('[Fe(H2O)6][Fe(CN)6]..19H2O(s)', '[Fe(H₂O)₆][Fe(CN)₆]·19H₂O(s)'), ('[Fe(H2O)6][Fe(CN)6]..19H2O(aq)', '[Fe(H₂O)₆][Fe(CN)₆]·19H₂O(aq)'), ('[Fe(CN)6]-3', '[Fe(CN)₆]³⁻'), ('[Fe(CN)6]-3(aq)', '[Fe(CN)₆]³⁻(aq)')])
@requires(parsing_library)
def test_formula_to_unicode(species, unicode):
    assert formula_to_unicode(species) == unicode