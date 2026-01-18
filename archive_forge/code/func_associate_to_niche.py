import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
import numpy
def associate_to_niche(fitnesses, reference_points, best_point, intercepts):
    """Associates individuals to reference points and calculates niche number.
    Corresponds to Algorithm 3 of Deb & Jain (2014)."""
    fn = (fitnesses - best_point) / (intercepts - best_point + numpy.finfo(float).eps)
    fn = numpy.repeat(numpy.expand_dims(fn, axis=1), len(reference_points), axis=1)
    norm = numpy.linalg.norm(reference_points, axis=1)
    distances = numpy.sum(fn * reference_points, axis=2) / norm.reshape(1, -1)
    distances = distances[:, :, numpy.newaxis] * reference_points[numpy.newaxis, :, :] / norm[numpy.newaxis, :, numpy.newaxis]
    distances = numpy.linalg.norm(distances - fn, axis=2)
    niches = numpy.argmin(distances, axis=1)
    distances = distances[list(range(niches.shape[0])), niches]
    return (niches, distances)