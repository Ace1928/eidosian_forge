from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def resolve_overload(self, key, cases, args, kws, allow_ambiguous=True, unsafe_casting=True, exact_match_required=False):
    """
        Given actual *args* and *kws*, find the best matching
        signature in *cases*, or None if none matches.
        *key* is used for error reporting purposes.
        If *allow_ambiguous* is False, a tie in the best matches
        will raise an error.
        If *unsafe_casting* is False, unsafe casting is forbidden.
        """
    assert not kws, 'Keyword arguments are not supported, yet'
    options = {'unsafe_casting': unsafe_casting, 'exact_match_required': exact_match_required}
    candidates = []
    for case in cases:
        if len(args) == len(case.args):
            rating = self._rate_arguments(args, case.args, **options)
            if rating is not None:
                candidates.append((rating.astuple(), case))
    candidates.sort(key=lambda i: i[0])
    if candidates:
        best_rate, best = candidates[0]
        if not allow_ambiguous:
            tied = []
            for rate, case in candidates:
                if rate != best_rate:
                    break
                tied.append(case)
            if len(tied) > 1:
                args = (key, args, '\n'.join(map(str, tied)))
                msg = 'Ambiguous overloading for %s %s:\n%s' % args
                raise TypeError(msg)
        return best