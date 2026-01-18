import os
import shutil
import subprocess
import tempfile
from ..printing import latex
from ..kinetics.rates import RadiolyticBase
from ..units import to_unitless, get_derived_unit
def rsys2tablines(rsys, rref0=1, coldelim=' & ', tex=True, ref_fmt=None, unit_registry=None, unit_fmt='{}', k_fmt='%.4g'):
    """
    Generates a table representation of a ReactionSystem.

    Parameters
    ----------
    rsys : ReactionSystem
    rref0 : integer
        default start of index counter (default: 1)
    coldelim : string
        column delimiter (default: ' & ')
    tex : bool
        use latex formatted output (default: True)
    ref_fmt : string or callable
        format string of ``ref`` attribute of reactions
    unit_registry : unit registry
        optional (default: None)
    """
    if ref_fmt is None:

        def _doi(s):
            return '\\texttt{\\href{http://dx.doi.org/' + s + '}{doi:' + s + '}}'

        def ref_fmt(s):
            if s is None:
                return 'None'
            if tex:
                if isinstance(s, dict):
                    return _doi(s['doi'])
                if s.startswith('doi:'):
                    return _doi(s[4:])
            return s

    def _wrap(s):
        if tex:
            return '\\ensuremath{' + s + '}'
        else:
            return s
    lines = []
    for ri, rxn in enumerate(rsys.rxns):
        rxn_ref = rxn.ref
        if isinstance(rxn.param, RadiolyticBase):
            if unit_registry is not None:
                kunit = get_derived_unit(unit_registry, 'radiolytic_yield')
                k = k_fmt % to_unitless(rxn.param.args[0], kunit)
                k_unit_str = kunit.dimensionality.latex.strip('$') if tex else kunit.dimensionality
        elif unit_registry is not None:
            kunit = get_derived_unit(unit_registry, 'concentration') ** (1 - rxn.order()) / get_derived_unit(unit_registry, 'time')
            try:
                k = k_fmt % to_unitless(rxn.param, kunit)
                k_unit_str = kunit.dimensionality.latex.strip('$') if tex else kunit.dimensionality
            except Exception:
                k, k_unit_str = rxn.param.equation_as_string(k_fmt, tex)
        else:
            k_unit_str = '-'
            if isinstance(k_fmt, str):
                k = k_fmt % rxn.param
            else:
                k = k_fmt(rxn.param)
        latex_kw = dict(with_param=False, with_name=False)
        if tex:
            latex_kw['substances'] = rsys.substances
            latex_kw['Reaction_around_arrow'] = ('}}' + coldelim + '\\ensuremath{{', '}}' + coldelim + '\\ensuremath{{')
        else:
            latex_kw['Reaction_around_arrow'] = (coldelim,) * 2
            latex_kw['Reaction_arrow'] = '->'
        lines.append(coldelim.join([str(rref0 + ri), ('\\ensuremath{%s}' if tex else '%s') % latex(rxn, **latex_kw), _wrap(k), unit_fmt.format(_wrap(k_unit_str)), ref_fmt(rxn_ref) if callable(ref_fmt) else ref_fmt.format(rxn_ref)]))
    return lines