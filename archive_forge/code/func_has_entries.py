from typing import Any, Hashable, Mapping, Optional, TypeVar, Union, overload
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
def has_entries(*keys_valuematchers, **kv_args):
    """Matches if dictionary contains entries satisfying a dictionary of keys
    and corresponding value matchers.

    :param matcher_dict: A dictionary mapping keys to associated value matchers,
        or to expected values for
        :py:func:`~hamcrest.core.core.isequal.equal_to` matching.

    Note that the keys must be actual keys, not matchers. Any value argument
    that is not a matcher is implicitly wrapped in an
    :py:func:`~hamcrest.core.core.isequal.equal_to` matcher to check for
    equality.

    Examples::

        has_entries({'foo':equal_to(1), 'bar':equal_to(2)})
        has_entries({'foo':1, 'bar':2})

    ``has_entries`` also accepts a list of keyword arguments:

    .. function:: has_entries(keyword1=value_matcher1[, keyword2=value_matcher2[, ...]])

    :param keyword1: A keyword to look up.
    :param valueMatcher1: The matcher to satisfy for the value, or an expected
        value for :py:func:`~hamcrest.core.core.isequal.equal_to` matching.

    Examples::

        has_entries(foo=equal_to(1), bar=equal_to(2))
        has_entries(foo=1, bar=2)

    Finally, ``has_entries`` also accepts a list of alternating keys and their
    value matchers:

    .. function:: has_entries(key1, value_matcher1[, ...])

    :param key1: A key (not a matcher) to look up.
    :param valueMatcher1: The matcher to satisfy for the value, or an expected
        value for :py:func:`~hamcrest.core.core.isequal.equal_to` matching.

    Examples::

        has_entries('foo', equal_to(1), 'bar', equal_to(2))
        has_entries('foo', 1, 'bar', 2)

    """
    if len(keys_valuematchers) == 1:
        try:
            base_dict = keys_valuematchers[0].copy()
            for key in base_dict:
                base_dict[key] = wrap_matcher(base_dict[key])
        except AttributeError:
            raise ValueError('single-argument calls to has_entries must pass a dict as the argument')
    else:
        if len(keys_valuematchers) % 2:
            raise ValueError('has_entries requires key-value pairs')
        base_dict = {}
        for index in range(int(len(keys_valuematchers) / 2)):
            base_dict[keys_valuematchers[2 * index]] = wrap_matcher(keys_valuematchers[2 * index + 1])
    for key, value in kv_args.items():
        base_dict[key] = wrap_matcher(value)
    return IsDictContainingEntries(base_dict)