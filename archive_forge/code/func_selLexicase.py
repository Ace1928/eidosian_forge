import random
import numpy as np
from functools import partial
from operator import attrgetter
def selLexicase(individuals, k):
    """Returns an individual that does the best on the fitness cases when
    considered one at a time in random order.
    http://faculty.hampshire.edu/lspector/pubs/lexicase-IEEE-TEC.pdf

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    """
    selected_individuals = []
    for i in range(k):
        fit_weights = individuals[0].fitness.weights
        candidates = individuals
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)
        while len(cases) > 0 and len(candidates) > 1:
            f = max if fit_weights[cases[0]] > 0 else min
            best_val_for_case = f((x.fitness.values[cases[0]] for x in candidates))
            candidates = [x for x in candidates if x.fitness.values[cases[0]] == best_val_for_case]
            cases.pop(0)
        selected_individuals.append(random.choice(candidates))
    return selected_individuals