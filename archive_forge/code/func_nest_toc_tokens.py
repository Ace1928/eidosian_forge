from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue, AMP_SUBSTITUTE, deprecated, HTML_PLACEHOLDER_RE, AtomicString
from ..treeprocessors import UnescapeTreeprocessor
from ..serializers import RE_AMP
import re
import html
import unicodedata
from copy import deepcopy
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any, Iterator, MutableSet
def nest_toc_tokens(toc_list):
    """Given an unsorted list with errors and skips, return a nested one.

        [{'level': 1}, {'level': 2}]
        =>
        [{'level': 1, 'children': [{'level': 2, 'children': []}]}]

    A wrong list is also converted:

        [{'level': 2}, {'level': 1}]
        =>
        [{'level': 2, 'children': []}, {'level': 1, 'children': []}]
    """
    ordered_list = []
    if len(toc_list):
        last = toc_list.pop(0)
        last['children'] = []
        levels = [last['level']]
        ordered_list.append(last)
        parents = []
        while toc_list:
            t = toc_list.pop(0)
            current_level = t['level']
            t['children'] = []
            if current_level < levels[-1]:
                levels.pop()
                to_pop = 0
                for p in reversed(parents):
                    if current_level <= p['level']:
                        to_pop += 1
                    else:
                        break
                if to_pop:
                    levels = levels[:-to_pop]
                    parents = parents[:-to_pop]
                levels.append(current_level)
            if current_level == levels[-1]:
                (parents[-1]['children'] if parents else ordered_list).append(t)
            else:
                last['children'].append(t)
                parents.append(last)
                levels.append(current_level)
            last = t
    return ordered_list