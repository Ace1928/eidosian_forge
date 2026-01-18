from collections import OrderedDict
from chempy import Reaction
from chempy.kinetics.rates import MassAction, RadiolyticBase
from chempy.units import to_unitless, default_units as u
def render_setup(self, *, ics, atol, tex=True, tspan=None):
    export = ''
    export += 'p = %s\n' % jl_dict(self.pars)
    export += 'ics = %s\n' % jl_dict(OrderedDict({self.substance_key_map[k]: v for k, v in to_unitless(ics, u.molar).items()}))
    if atol:
        export += 'abstol_d = %s\n' % jl_dict({self.substance_key_map[k]: v for k, v in to_unitless(atol, u.molar).items()})
        export += 'abstol = Array([get(abstol_d, k, 1e-10) for k=keys(speciesmap(rn))])'
    if tex:
        export += 'subst_tex = Dict([%s])\n' % ', '.join(('(:%s, ("%s", "%s"))' % (v, k, k.latex_name) for k, v in self.substance_key_map.items()))
    if tspan:
        export += 'tspan = (0., %12.5g)\nu0 = Array([get(ics, k, 1e-28) for k=keys(speciesmap(rn))])\nparr = Array([p[k] for k=keys(paramsmap(rn))])\noprob = ODEProblem(rn, u0, tspan, parr)\n'
    return export