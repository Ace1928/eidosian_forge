import re
from .utilities import MethodMappingList
from .component import NonZeroDimensionalComponent
from .coordinates import PtolemyCoordinates
from .rur import RUR
from . import processFileBase
from ..pari import pari
def parse_rs_rur(text, variables):
    m = re.match('\\[([^,\\]]+),\\s*([^,\\]]+),\\s*\\[([^\\]]+)\\]\\s*,\\s*\\[\\s*\\]\\s*\\]', text, re.DOTALL)
    if not m:
        raise Exception('Format not detected')
    extension_str, denominator_str, numerators_str = m.groups()
    var = _find_var_of_poly(extension_str)
    extension = pari(extension_str.replace(var, 'x'))
    denominator = pari(denominator_str.replace(var, 'x'))
    numerators = [pari(numerator_str.replace(var, 'x')) for numerator_str in numerators_str.split(',')]
    fracs = [RUR([(numerator.Mod(extension), 1), (denominator.Mod(extension), -1)]) for numerator in numerators]
    return dict(list(zip(variables, fracs)) + [('1', RUR.from_int(1))])