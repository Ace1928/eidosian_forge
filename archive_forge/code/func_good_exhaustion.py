from .links_base import Strand, Crossing, Link
import random
import collections
def good_exhaustion(link, max_tries=20):
    """
    Uses a random search to try to find an Exhaustion with small width.

    >>> ge = good_exhaustion(Link('K4a1'))
    >>> ge.width
    2
    """
    E_best = None
    crossings = list(link.crossings)
    tries = 0
    while tries < max_tries:
        random.shuffle(crossings)
        for C in crossings:
            E = MorseExhaustion(link, C)
            if E_best is None or E.width < E_best.width:
                E_best = E
            tries += 1
            if tries >= max_tries:
                break
    return E_best