from __future__ import annotations
import re
from fractions import Fraction
def disordered_formula(disordered_struct, symbols=('x', 'y', 'z'), fmt='plain'):
    """Returns a formula of a form like AxB1-x (x=0.5)
    for disordered structures. Will only return a
    formula for disordered structures with one
    kind of disordered site at present.

    Args:
        disordered_struct: a disordered structure
        symbols: a tuple of characters to use for
        subscripts, by default this is ('x', 'y', 'z')
        but if you have more than three disordered
        species more symbols will need to be added
        fmt (str): 'plain', 'HTML' or 'LaTeX'

    Returns:
        str: a disordered formula string
    """
    from pymatgen.core import Composition, get_el_sp
    if disordered_struct.is_ordered:
        raise ValueError('Structure is not disordered, so disordered formula not defined.')
    disordered_site_compositions = {site.species for site in disordered_struct if not site.is_ordered}
    if len(disordered_site_compositions) > 1:
        raise ValueError('Ambiguous how to define disordered formula when more than one type of disordered site is present.')
    disordered_site_composition = disordered_site_compositions.pop()
    disordered_species = {str(sp) for sp, occu in disordered_site_composition.items()}
    if len(disordered_species) > len(symbols):
        raise ValueError(f'Not enough symbols to describe disordered composition: {symbols}')
    symbols = list(symbols)[0:len(disordered_species) - 1]
    comp = disordered_struct.composition.get_el_amt_dict().items()
    comp = sorted(comp, key=lambda x: get_el_sp(x[0]).X)
    disordered_comp = []
    variable_map = {}
    total_disordered_occu = sum((occu for sp, occu in comp if str(sp) in disordered_species))
    factor_comp = disordered_struct.composition.as_dict()
    factor_comp['X'] = total_disordered_occu
    for sp in disordered_species:
        del factor_comp[str(sp)]
    factor_comp = Composition.from_dict(factor_comp)
    factor = factor_comp.get_reduced_formula_and_factor()[1]
    total_disordered_occu /= factor
    remainder = f'{formula_double_format(total_disordered_occu, ignore_ones=False)}-{'-'.join(symbols)}'
    for sp, occu in comp:
        species = str(sp)
        if species not in disordered_species:
            disordered_comp.append((species, formula_double_format(occu / factor)))
        elif len(symbols) > 0:
            symbol = symbols.pop(0)
            disordered_comp.append((species, symbol))
            variable_map[symbol] = occu / total_disordered_occu / factor
        else:
            disordered_comp.append((species, remainder))
    if fmt == 'LaTeX':
        sub_start = '_{'
        sub_end = '}'
    elif fmt == 'HTML':
        sub_start = '<sub>'
        sub_end = '</sub>'
    elif fmt != 'plain':
        raise ValueError('Unsupported output format, choose from: LaTeX, HTML, plain')
    disordered_formula = []
    for sp, occu in disordered_comp:
        disordered_formula.append(sp)
        if occu:
            if fmt != 'plain':
                disordered_formula.append(sub_start)
            disordered_formula.append(occu)
            if fmt != 'plain':
                disordered_formula.append(sub_end)
    disordered_formula.append(' ')
    disordered_formula += [f'{key}={formula_double_format(val)} ' for key, val in variable_map.items()]
    return ''.join(map(str, disordered_formula))[0:-1]