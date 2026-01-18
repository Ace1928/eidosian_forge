import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def categorize_substances(self, **kwargs):
    """Returns categories of substance keys (e.g. nonparticipating, unaffected etc.)

        Some substances are only *accumulated* (i.e. irreversibly formed) and are never net
        reactants in any reactions, others are *depleted* (they are never net proucts in
        any reaction). Some substanaces are *unaffected* since they appear with equal coefficients on
        both reactant and product side, while some may be *nonparticipating* (they don't appear on
        either side and have thus no effect on the reactionsystem).

        Parameters
        ----------
        \\*\\*kwargs:
             Keyword arguments passed on to :class:`ReactionSystem`.

        Returns
        -------
        dict of sets of substance keys, the dictionary has the following keys:
            - ``'accumulated'``: keys only participating as net products.
            - ``'depleted'``: keys only participating as net reactants.
            - ``'unaffected'``: keys appearing in reactions but with zero net effect.
            - ``'nonparticipating'``: keys not appearing in any reactions.

        """
    import numpy as np
    irrev_rxns = []
    for r in self.rxns:
        try:
            irrev_rxns.extend(r.as_reactions())
        except AttributeError:
            irrev_rxns.append(r)
    irrev_rsys = ReactionSystem(irrev_rxns, self.substances, **kwargs)
    all_r = irrev_rsys.all_reac_stoichs()
    all_p = irrev_rsys.all_prod_stoichs()
    if np.any(all_r < 0) or np.any(all_p < 0):
        raise ValueError('Expected positive stoichiometric coefficients')
    net = all_p - all_r
    accumulated, depleted, unaffected, nonparticipating = (set(), set(), set(), set())
    for i, sk in enumerate(irrev_rsys.substances.keys()):
        in_r = np.any(net[:, i] < 0)
        in_p = np.any(net[:, i] > 0)
        if in_r and in_p:
            pass
        elif in_r:
            depleted.add(sk)
        elif in_p:
            accumulated.add(sk)
        elif np.any(all_p[:, i] > 0):
            assert np.all(all_p[:, i] == all_r[:, i]), 'Open issue at github.com/bjodah/chempy'
            unaffected.add(sk)
        else:
            nonparticipating.add(sk)
    return dict(accumulated=accumulated, depleted=depleted, unaffected=unaffected, nonparticipating=nonparticipating)