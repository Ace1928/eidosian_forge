import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
import numpy
def niching(individuals, k, niches, distances, niche_counts):
    selected = []
    available = numpy.ones(len(individuals), dtype=bool)
    while len(selected) < k:
        n = k - len(selected)
        available_niches = numpy.zeros(len(niche_counts), dtype=bool)
        available_niches[numpy.unique(niches[available])] = True
        min_count = numpy.min(niche_counts[available_niches])
        selected_niches = numpy.flatnonzero(numpy.logical_and(available_niches, niche_counts == min_count))
        numpy.random.shuffle(selected_niches)
        selected_niches = selected_niches[:n]
        for niche in selected_niches:
            niche_individuals = numpy.flatnonzero(numpy.logical_and(niches == niche, available))
            numpy.random.shuffle(niche_individuals)
            if niche_counts[niche] == 0:
                sel_index = niche_individuals[numpy.argmin(distances[niche_individuals])]
            else:
                sel_index = niche_individuals[0]
            available[sel_index] = False
            niche_counts[niche] += 1
            selected.append(individuals[sel_index])
    return selected