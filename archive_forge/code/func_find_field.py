from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
def find_field(self, prec, degree, optimize=False, verbosity=False):
    if self._field[False] is None:
        self._field[False] = self._find_field_uncached(prec, degree, verbosity)
    if optimize and self._field[True] is None:
        if self._field[False] is None:
            return None
        K, z, exact_elts = self._field[False]
        z = optimize_field_generator(z)
        exact_elts = self.are_in_field_generated_by(z, prec)
        if exact_elts is None:
            if verbosity:
                print('Bailing: Could not express elements in optimized generator', z)
            return None
        field = z.number_field()
        exact_elts = [field(exact_elt) for exact_elt in exact_elts]
        self._field[True] = (field, z, exact_elts)
    return self._field[optimize]