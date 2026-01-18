from __future__ import unicode_literals
import functools
import re
from datetime import timedelta
import logging
import io
def sort_and_reindex(subtitles, start_index=1, in_place=False, skip=True):
    """
    Reorder subtitles to be sorted by start time order, and rewrite the indexes
    to be in that same order. This ensures that the SRT file will play in an
    expected fashion after, for example, times were changed in some subtitles
    and they may need to be resorted.

    If skip=True, subtitles will also be skipped if they are considered not to
    be useful. Currently, the conditions to be considered "not useful" are as
    follows:

    - Content is empty, or only whitespace
    - The start time is negative
    - The start time is equal to or later than the end time

    .. doctest::

        >>> from datetime import timedelta
        >>> one = timedelta(seconds=1)
        >>> two = timedelta(seconds=2)
        >>> three = timedelta(seconds=3)
        >>> subs = [
        ...     Subtitle(index=999, start=one, end=two, content='1'),
        ...     Subtitle(index=0, start=two, end=three, content='2'),
        ... ]
        >>> list(sort_and_reindex(subs))  # doctest: +ELLIPSIS
        [Subtitle(...index=1...), Subtitle(...index=2...)]

    :param subtitles: :py:class:`Subtitle` objects in any order
    :param int start_index: The index to start from
    :param bool in_place: Whether to modify subs in-place for performance
                          (version <=1.0.0 behaviour)
    :param bool skip: Whether to skip subtitles considered not useful (see
                      above for rules)
    :returns: The sorted subtitles
    :rtype: :term:`generator` of :py:class:`Subtitle` objects
    """
    skipped_subs = 0
    for sub_num, subtitle in enumerate(sorted(subtitles), start=start_index):
        if not in_place:
            subtitle = Subtitle(**vars(subtitle))
        if skip:
            try:
                _should_skip_sub(subtitle)
            except _ShouldSkipException as thrown_exc:
                if subtitle.index is None:
                    LOG.info('Skipped subtitle with no index: %s', thrown_exc)
                else:
                    LOG.info('Skipped subtitle at index %d: %s', subtitle.index, thrown_exc)
                skipped_subs += 1
                continue
        subtitle.index = sub_num - skipped_subs
        yield subtitle