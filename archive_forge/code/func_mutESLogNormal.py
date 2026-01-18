import math
import random
from itertools import repeat
def mutESLogNormal(individual, c, indpb):
    """Mutate an evolution strategy according to its :attr:`strategy`
    attribute as described in [Beyer2002]_. First the strategy is mutated
    according to an extended log normal rule, :math:`\\\\boldsymbol{\\sigma}_t =
    \\\\exp(\\\\tau_0 \\mathcal{N}_0(0, 1)) \\\\left[ \\\\sigma_{t-1, 1}\\\\exp(\\\\tau
    \\mathcal{N}_1(0, 1)), \\ldots, \\\\sigma_{t-1, n} \\\\exp(\\\\tau
    \\mathcal{N}_n(0, 1))\\\\right]`, with :math:`\\\\tau_0 =
    \\\\frac{c}{\\\\sqrt{2n}}` and :math:`\\\\tau = \\\\frac{c}{\\\\sqrt{2\\\\sqrt{n}}}`,
    the the individual is mutated by a normal distribution of mean 0 and
    standard deviation of :math:`\\\\boldsymbol{\\sigma}_{t}` (its current
    strategy) then . A recommended choice is ``c=1`` when using a :math:`(10,
    100)` evolution strategy [Beyer2002]_ [Schwefel1995]_.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param c: The learning parameter.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.

    .. [Beyer2002] Beyer and Schwefel, 2002, Evolution strategies - A
       Comprehensive Introduction

    .. [Schwefel1995] Schwefel, 1995, Evolution and Optimum Seeking.
       Wiley, New York, NY
    """
    size = len(individual)
    t = c / math.sqrt(2.0 * math.sqrt(size))
    t0 = c / math.sqrt(2.0 * size)
    n = random.gauss(0, 1)
    t0_n = t0 * n
    for indx in range(size):
        if random.random() < indpb:
            individual.strategy[indx] *= math.exp(t0_n + t * random.gauss(0, 1))
            individual[indx] += individual.strategy[indx] * random.gauss(0, 1)
    return (individual,)