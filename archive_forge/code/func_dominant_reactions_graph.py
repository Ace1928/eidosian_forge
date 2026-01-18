from .. import ReactionSystem, Equilibrium
from ..units import get_derived_unit, to_unitless, default_units as u
def dominant_reactions_graph(concs, rate_exprs_cb, rsys, substance_key, linthreshy=1e-09, fname='dominant_reactions_graph.png', relative=False, combine_equilibria=False, **kwargs):
    from ..util.graph import rsys2graph
    rates = rate_exprs_cb(0, concs)
    eqk1, eqk2, eqs = _combine_rxns_to_eq(rsys) if combine_equilibria else ([], [], [])
    rrate, rxns = zip(*_dominant_reaction_effects(substance_key, rsys, rates, linthreshy, eqk1, eqk2, eqs)[0])
    rsys = ReactionSystem(rxns, rsys.substances, rsys.name)
    lg_rrate = np.log10(np.abs(rrate))
    rsys2graph(rsys, fname=fname, penwidths=1 + lg_rrate - np.min(lg_rrate), **kwargs)