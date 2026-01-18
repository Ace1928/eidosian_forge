import unicodedata
from pyglet.gl import *
from pyglet import image
def get_grapheme_clusters(text):
    """Implements Table 2 of UAX #29: Grapheme Cluster Boundaries.

    Does not currently implement Hangul syllable rules.
    
    :Parameters:
        `text` : unicode
            String to cluster.

    .. versionadded:: 1.1.2

    :rtype: List of `unicode`
    :return: List of Unicode grapheme clusters
    """
    clusters = []
    cluster = ''
    left = None
    for right in text:
        if cluster and grapheme_break(left, right):
            clusters.append(cluster)
            cluster = ''
        elif cluster:
            clusters.append(u'\u200b')
        cluster += right
        left = right
    if cluster:
        clusters.append(cluster)
    return clusters