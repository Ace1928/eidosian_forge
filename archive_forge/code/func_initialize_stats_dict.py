import numpy as np
from deap import tools, gp
from inspect import isclass
from .operator_utils import set_sample_weight
from sklearn.utils import indexable
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from collections import defaultdict
import warnings
from stopit import threading_timeoutable, TimeoutException
def initialize_stats_dict(individual):
    """
    Initializes the stats dict for individual
    The statistics initialized are:
        'generation': generation in which the individual was evaluated. Initialized as: 0
        'mutation_count': number of mutation operations applied to the individual and its predecessor cumulatively. Initialized as: 0
        'crossover_count': number of crossover operations applied to the individual and its predecessor cumulatively. Initialized as: 0
        'predecessor': string representation of the individual. Initialized as: ('ROOT',)

    Parameters
    ----------
    individual: deap individual

    Returns
    -------
    object
    """
    individual.statistics['generation'] = 0
    individual.statistics['mutation_count'] = 0
    individual.statistics['crossover_count'] = 0
    individual.statistics['predecessor'] = ('ROOT',)