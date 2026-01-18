import random
import numpy as np
from functools import partial
from operator import attrgetter
def selDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first, fit_attr='fitness'):
    """Tournament selection which use the size of the individuals in order
    to discriminate good solutions. This kind of tournament is obviously
    useless with fixed-length representation, but has been shown to
    significantly reduce excessive growth of individuals, especially in GP,
    where it can be used as a bloat control technique (see
    [Luke2002fighting]_). This selection operator implements the double
    tournament technique presented in this paper.

    The core principle is to use a normal tournament selection, but using a
    special sample function to select aspirants, which is another tournament
    based on the size of the individuals. To ensure that the selection
    pressure is not too high, the size of the size tournament (the number
    of candidates evaluated) can be a real number between 1 and 2. In this
    case, the smaller individual among two will be selected with a probability
    *size_tourn_size*/2. For instance, if *size_tourn_size* is set to 1.4,
    then the smaller individual will have a 0.7 probability to be selected.

    .. note::
        In GP, it has been shown that this operator produces better results
        when it is combined with some kind of a depth limit.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fitness_size: The number of individuals participating in each     fitness tournament
    :param parsimony_size: The number of individuals participating in each     size tournament. This value has to be a real number    in the range [1,2], see above for details.
    :param fitness_first: Set this to True if the first tournament done should     be the fitness one (i.e. the fitness tournament producing aspirants for     the size tournament). Setting it to False will behaves as the opposite     (size tournament feeding fitness tournaments with candidates). It has been     shown that this parameter does not have a significant effect in most cases    (see [Luke2002fighting]_).
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    .. [Luke2002fighting] Luke and Panait, 2002, Fighting bloat with
        nonparametric parsimony pressure
    """
    assert 1 <= parsimony_size <= 2, 'Parsimony tournament size has to be in the range [1, 2].'

    def _sizeTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            prob = parsimony_size / 2.0
            ind1, ind2 = select(individuals, k=2)
            if len(ind1) > len(ind2):
                ind1, ind2 = (ind2, ind1)
            elif len(ind1) == len(ind2):
                prob = 0.5
            chosen.append(ind1 if random.random() < prob else ind2)
        return chosen

    def _fitTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            aspirants = select(individuals, k=fitness_size)
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen
    if fitness_first:
        tfit = partial(_fitTournament, select=selRandom)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=selRandom)
        return _fitTournament(individuals, k, tsize)