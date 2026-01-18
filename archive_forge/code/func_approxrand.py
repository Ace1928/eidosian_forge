import operator
from functools import reduce
from math import fabs
from random import shuffle
from nltk.util import LazyConcatenation, LazyMap
def approxrand(a, b, **kwargs):
    """
    Returns an approximate significance level between two lists of
    independently generated test values.

    Approximate randomization calculates significance by randomly drawing
    from a sample of the possible permutations. At the limit of the number
    of possible permutations, the significance level is exact. The
    approximate significance level is the sample mean number of times the
    statistic of the permutated lists varies from the actual statistic of
    the unpermuted argument lists.

    :return: a tuple containing an approximate significance level, the count
             of the number of times the pseudo-statistic varied from the
             actual statistic, and the number of shuffles
    :rtype: tuple
    :param a: a list of test values
    :type a: list
    :param b: another list of independently generated test values
    :type b: list
    """
    shuffles = kwargs.get('shuffles', 999)
    shuffles = min(shuffles, reduce(operator.mul, range(1, len(a) + len(b) + 1)))
    stat = kwargs.get('statistic', lambda lst: sum(lst) / len(lst))
    verbose = kwargs.get('verbose', False)
    if verbose:
        print('shuffles: %d' % shuffles)
    actual_stat = fabs(stat(a) - stat(b))
    if verbose:
        print('actual statistic: %f' % actual_stat)
        print('-' * 60)
    c = 1e-100
    lst = LazyConcatenation([a, b])
    indices = list(range(len(a) + len(b)))
    for i in range(shuffles):
        if verbose and i % 10 == 0:
            print('shuffle: %d' % i)
        shuffle(indices)
        pseudo_stat_a = stat(LazyMap(lambda i: lst[i], indices[:len(a)]))
        pseudo_stat_b = stat(LazyMap(lambda i: lst[i], indices[len(a):]))
        pseudo_stat = fabs(pseudo_stat_a - pseudo_stat_b)
        if pseudo_stat >= actual_stat:
            c += 1
        if verbose and i % 10 == 0:
            print('pseudo-statistic: %f' % pseudo_stat)
            print('significance: %f' % ((c + 1) / (i + 1)))
            print('-' * 60)
    significance = (c + 1) / (shuffles + 1)
    if verbose:
        print('significance: %f' % significance)
        if betai:
            for phi in [0.01, 0.05, 0.1, 0.15, 0.25, 0.5]:
                print(f'prob(phi<={phi:f}): {betai(c, shuffles, phi):f}')
    return (significance, c, shuffles)