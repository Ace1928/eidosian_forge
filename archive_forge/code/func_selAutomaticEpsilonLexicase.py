import random
import numpy as np
from functools import partial
from operator import attrgetter
def selAutomaticEpsilonLexicase(individuals, k):
    """
    Returns an individual that does the best on the fitness cases when considered one at a
    time in random order.
    https://push-language.hampshire.edu/uploads/default/original/1X/35c30e47ef6323a0a949402914453f277fb1b5b0.pdf
    Implemented lambda_epsilon_y implementation.

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
            errors_for_this_case = [x.fitness.values[cases[0]] for x in candidates]
            median_val = np.median(errors_for_this_case)
            median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case])
            if fit_weights[cases[0]] > 0:
                best_val_for_case = max(errors_for_this_case)
                min_val_to_survive = best_val_for_case - median_absolute_deviation
                candidates = [x for x in candidates if x.fitness.values[cases[0]] >= min_val_to_survive]
            else:
                best_val_for_case = min(errors_for_this_case)
                max_val_to_survive = best_val_for_case + median_absolute_deviation
                candidates = [x for x in candidates if x.fitness.values[cases[0]] <= max_val_to_survive]
            cases.pop(0)
        selected_individuals.append(random.choice(candidates))
    return selected_individuals