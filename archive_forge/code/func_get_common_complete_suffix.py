from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
def get_common_complete_suffix(document, completions):
    """
    Return the common prefix for all completions.
    """

    def doesnt_change_before_cursor(completion):
        end = completion.text[:-completion.start_position]
        return document.text_before_cursor.endswith(end)
    completions2 = [c for c in completions if doesnt_change_before_cursor(c)]
    if len(completions2) != len(completions):
        return ''

    def get_suffix(completion):
        return completion.text[-completion.start_position:]
    return _commonprefix([get_suffix(c) for c in completions2])