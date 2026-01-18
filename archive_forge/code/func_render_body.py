from collections import OrderedDict
from chempy import Reaction
from chempy.kinetics.rates import MassAction, RadiolyticBase
from chempy.units import to_unitless, default_units as u
def render_body(self, sparse_jac=False):
    name = 'rn'
    return self._template_body.format(crn_macro='min_reaction_network' if sparse_jac else 'reaction_network', name=name, reactions='\n    '.join(self.rxs), parameters=' '.join(self.pars), post='addodes!({}, sparse_jac=True)'.format(name) if sparse_jac else '')