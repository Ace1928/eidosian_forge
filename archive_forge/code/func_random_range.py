import abc
from Cryptodome.Util.py3compat import iter_range, bord, bchr, ABC
from Cryptodome import Random
@classmethod
def random_range(cls, **kwargs):
    """Generate a random integer within a given internal.

        :Keywords:
          min_inclusive : integer
            The lower end of the interval (inclusive).
          max_inclusive : integer
            The higher end of the interval (inclusive).
          max_exclusive : integer
            The higher end of the interval (exclusive).
          randfunc : callable
            A function that returns a random byte string. The length of the
            byte string is passed as parameter. Optional.
            If not provided (or ``None``), randomness is read from the system RNG.
        :Returns:
            An Integer randomly taken in the given interval.
        """
    min_inclusive = kwargs.pop('min_inclusive', None)
    max_inclusive = kwargs.pop('max_inclusive', None)
    max_exclusive = kwargs.pop('max_exclusive', None)
    randfunc = kwargs.pop('randfunc', None)
    if kwargs:
        raise ValueError('Unknown keywords: ' + str(kwargs.keys))
    if None not in (max_inclusive, max_exclusive):
        raise ValueError('max_inclusive and max_exclusive cannot be both specified')
    if max_exclusive is not None:
        max_inclusive = max_exclusive - 1
    if None in (min_inclusive, max_inclusive):
        raise ValueError('Missing keyword to identify the interval')
    if randfunc is None:
        randfunc = Random.new().read
    norm_maximum = max_inclusive - min_inclusive
    bits_needed = cls(norm_maximum).size_in_bits()
    norm_candidate = -1
    while not 0 <= norm_candidate <= norm_maximum:
        norm_candidate = cls.random(max_bits=bits_needed, randfunc=randfunc)
    return norm_candidate + min_inclusive