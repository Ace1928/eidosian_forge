from __future__ import absolute_import
from functools import reduce
import six
def fitness_and_quality_parsed(mime_type, parsed_ranges):
    """Find the best match for a mime-type amongst parsed media-ranges.

    Find the best match for a given mime-type against a list of media_ranges
    that have already been parsed by parse_media_range(). Returns a tuple of
    the fitness value and the value of the 'q' quality parameter of the best
    match, or (-1, 0) if no match was found. Just as for quality_parsed(),
    'parsed_ranges' must be a list of parsed media ranges.
    """
    best_fitness = -1
    best_fit_q = 0
    target_type, target_subtype, target_params = parse_media_range(mime_type)
    for type, subtype, params in parsed_ranges:
        type_match = type == target_type or type == '*' or target_type == '*'
        subtype_match = subtype == target_subtype or subtype == '*' or target_subtype == '*'
        if type_match and subtype_match:
            param_matches = reduce(lambda x, y: x + y, [1 for key, value in six.iteritems(target_params) if key != 'q' and key in params and (value == params[key])], 0)
            fitness = type == target_type and 100 or 0
            fitness += subtype == target_subtype and 10 or 0
            fitness += param_matches
            if fitness > best_fitness:
                best_fitness = fitness
                best_fit_q = params['q']
    return (best_fitness, float(best_fit_q))