import json
import re
from collections import (
from typing import (
import attr
from . import (
from .parsing import (
def str_search(self, search: str, include_persisted: bool=False) -> 'OrderedDict[int, HistoryItem]':
    """Find history items which contain a given string

        :param search: the string to search for
        :param include_persisted: if True, then search full history including persisted history
        :return: a dictionary of history items keyed by their 1-based index in ascending order,
                 or an empty dictionary if the string was not found
        """

    def isin(history_item: HistoryItem) -> bool:
        """filter function for string search of history"""
        sloppy = utils.norm_fold(search)
        inraw = sloppy in utils.norm_fold(history_item.raw)
        inexpanded = sloppy in utils.norm_fold(history_item.expanded)
        return inraw or inexpanded
    start = 0 if include_persisted else self.session_start_index
    return self._build_result_dictionary(start, len(self), isin)