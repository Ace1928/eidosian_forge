from collections import defaultdict
from os.path import join
from nltk.data import load
def map_tag(source, target, source_tag):
    """
    Maps the tag from the source tagset to the target tagset.

    >>> map_tag('en-ptb', 'universal', 'VBZ')
    'VERB'
    >>> map_tag('en-ptb', 'universal', 'VBP')
    'VERB'
    >>> map_tag('en-ptb', 'universal', '``')
    '.'
    """
    if target == 'universal':
        if source == 'wsj':
            source = 'en-ptb'
        if source == 'brown':
            source = 'en-brown'
    return tagset_mapping(source, target)[source_tag]