from typing import Collection, Dict, List, Union, overload, Iterable
from typing_extensions import Literal
import msgpack
from pkg_resources import resource_filename
import numpy as np
from numpy.typing import ArrayLike
from .._cache import cache
from .._typing import _FloatLike_co
@cache(level=10)
def plimit_intervals(*, primes: ArrayLike, bins_per_octave: int=12, sort: bool=True, return_factors: bool=False) -> Union[np.ndarray, List[Dict[int, int]]]:
    """Construct p-limit intervals for a given set of prime factors.

    This function is based on the "harmonic crystal growth" algorithm
    of [#1]_ [#2]_.

    .. [#1] Tenney, James.
        "On ‘Crystal Growth’ in harmonic space (1993–1998)."
        Contemporary Music Review 27.1 (2008): 47-56.

    .. [#2] Sabat, Marc, and James Tenney.
        "Three crystal growth algorithms in 23-limit constrained harmonic space."
        Contemporary Music Review 27, no. 1 (2008): 57-78.

    Parameters
    ----------
    primes : array of odd primes
        Which prime factors are to be used
    bins_per_octave : int
        The number of intervals to construct
    sort : bool
        If `True` then intervals are returned in ascending order.
        If `False`, then intervals are returned in crystal growth order.
    return_factors : bool
        If `True` then return a list of dictionaries encoding the prime factorization
        of each interval as `{2: p2, 3: p3, ...}` (meaning `3**p3 * 2**p2`).
        If `False` (default), return intervals as an array of floating point numbers.

    Returns
    -------
    intervals : np.ndarray or list of dictionaries
        The constructed interval set. All intervals are mapped
        to the range [1, 2).

    See Also
    --------
    pythagorean_intervals

    Examples
    --------
    Compare 3-limit tuning to Pythagorean tuning and 12-TET

    >>> librosa.plimit_intervals(primes=[3], bins_per_octave=12)
    array([1.        , 1.05349794, 1.125     , 1.18518519, 1.265625  ,
           1.33333333, 1.40466392, 1.5       , 1.58024691, 1.6875    ,
           1.77777778, 1.8984375 ])
    >>> # Pythagorean intervals:
    >>> librosa.pythagorean_intervals(bins_per_octave=12)
    array([1.        , 1.06787109, 1.125     , 1.20135498, 1.265625  ,
           1.35152435, 1.42382812, 1.5       , 1.60180664, 1.6875    ,
           1.80203247, 1.8984375 ])
    >>> # 12-TET intervals:
    >>> 2**(np.arange(12)/12)
    array([1.        , 1.05946309, 1.12246205, 1.18920712, 1.25992105,
           1.33483985, 1.41421356, 1.49830708, 1.58740105, 1.68179283,
           1.78179744, 1.88774863])

    Create a 7-bin, 5-limit interval set

    >>> librosa.plimit_intervals(primes=[3, 5], bins_per_octave=7)
    array([1.        , 1.125     , 1.25      , 1.33333333, 1.5       ,
           1.66666667, 1.875     ])

    The same example, but now in factored form

    >>> librosa.plimit_intervals(primes=[3, 5], bins_per_octave=7,
    ...                          return_factors=True)
    [
        {},
        {2: -3, 3: 2},
        {2: -2, 5: 1},
        {2: 2, 3: -1},
        {2: -1, 3: 1},
        {3: -1, 5: 1},
        {2: -3, 3: 1, 5: 1}
    ]
    """
    primes = np.atleast_1d(primes)
    logs = np.log2(primes, dtype=np.float64)
    seeds = []
    for i in range(len(primes)):
        seed = [0] * len(primes)
        seed[i] = 1
        seeds.append(tuple(seed))
        seed[i] = -1
        seeds.append(tuple(seed))
    frontier = seeds.copy()
    distances = dict()
    intervals = list()
    root = tuple([0] * len(primes))
    intervals.append(root)
    while len(intervals) < bins_per_octave:
        score = np.inf
        best_f = 0
        for f, point in enumerate(frontier):
            HD = 0.0
            for s in intervals:
                if (s, point) not in distances:
                    distances[s, point] = __harmonic_distance(logs, point, s)
                    distances[point, s] = distances[s, point]
                HD += distances[s, point]
            if HD < score or (np.isclose(HD, score) and _crystal_tie_break(point, frontier[best_f], logs)):
                score = HD
                best_f = f
        new_point = frontier.pop(best_f)
        intervals.append(new_point)
        for _ in seeds:
            new_seed = tuple(np.array(new_point) + np.array(_))
            if new_seed not in intervals and new_seed not in frontier:
                frontier.append(new_seed)
    pows = np.array(list(intervals), dtype=float)
    log_ratios: np.ndarray
    pow2: np.ndarray
    log_ratios, pow2 = np.modf(pows.dot(logs))
    too_small = log_ratios < 0
    log_ratios[too_small] += 1
    pow2[too_small] -= 1
    pow2 = pow2.astype(int)
    idx: Iterable[int]
    if sort:
        idx = np.argsort(log_ratios)
        log_ratios = log_ratios[idx]
    else:
        idx = range(bins_per_octave)
    if return_factors:
        factors = []
        for i in idx:
            v = dict()
            if pow2[i] != 0:
                v[2] = -pow2[i]
            v.update({p: int(power) for p, power in zip(primes, pows[i]) if power != 0})
            factors.append(v)
        return factors
    return np.power(2, log_ratios)