import pydoc
from contextlib import suppress
from typing import Dict, Optional
from jedi.inference.names import AbstractArbitraryName
def imitate_pydoc(string):
    """
    It's not possible to get the pydoc's without starting the annoying pager
    stuff.
    """
    if pydoc_topics is None:
        return ''
    h = pydoc.help
    with suppress(KeyError):
        string = h.symbols[string]
        string, _, related = string.partition(' ')

    def get_target(s):
        return h.topics.get(s, h.keywords.get(s))
    while isinstance(string, str):
        string = get_target(string)
    try:
        label, related = string
    except TypeError:
        return ''
    try:
        return pydoc_topics[label].strip() if pydoc_topics else ''
    except KeyError:
        return ''