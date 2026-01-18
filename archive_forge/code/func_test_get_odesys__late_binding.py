from collections import defaultdict, OrderedDict
from itertools import permutations
import math
import pytest
from chempy import Equilibrium, Reaction, ReactionSystem, Substance
from chempy.thermodynamics.expressions import MassActionEq
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.testing import requires
from .test_rates import _get_SpecialFraction_rsys
from ..arrhenius import ArrheniusParam
from ..rates import Arrhenius, MassAction, Radiolytic, RampedTemp
from .._rates import ShiftedTPoly
from ..ode import (
from ..integrated import dimerization_irrev, binary_rev
@requires('numpy', 'pyodesys')
def test_get_odesys__late_binding():

    def _gibbs(args, T, R, backend, **kwargs):
        H, S = args
        return backend.exp(-(H - T * S) / (R * T))

    def _eyring(args, T, R, k_B, h, backend, **kwargs):
        H, S = args
        return k_B / h * T * backend.exp(-(H - T * S) / (R * T))
    gibbs_pk = ('temperature', 'molar_gas_constant')
    eyring_pk = gibbs_pk + ('Boltzmann_constant', 'Planck_constant')
    GibbsEC = MassActionEq.from_callback(_gibbs, argument_names=('H', 'S'), parameter_keys=gibbs_pk)
    EyringMA = MassAction.from_callback(_eyring, argument_names=('H', 'S'), parameter_keys=eyring_pk)
    uk_equil = ('He_assoc', 'Se_assoc')
    beta = GibbsEC(unique_keys=uk_equil)
    uk_kinet = ('Ha_assoc', 'Sa_assoc')
    bimol_barrier = EyringMA(unique_keys=uk_kinet)
    eq = Equilibrium({'Fe+3', 'SCN-'}, {'FeSCN+2'}, beta)
    rsys = ReactionSystem(eq.as_reactions(kf=bimol_barrier))
    odesys, extra = get_odesys(rsys, include_params=False)
    pk, unique, p_units = map(extra.get, 'param_keys unique p_units'.split())
    assert sorted(unique) == sorted(uk_equil + uk_kinet)
    assert sorted(pk) == sorted(eyring_pk)