import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def fes(self, name=None, frame=None):
    """
        Lists frame element objects. If 'name' is provided, this is treated as
        a case-insensitive regular expression to filter by frame name.
        (Case-insensitivity is because casing of frame element names is not always
        consistent across frames.) Specify 'frame' to filter by a frame name pattern,
        ID, or object.

        >>> from nltk.corpus import framenet as fn
        >>> fn.fes('Noise_maker')
        [<fe ID=6043 name=Noise_maker>]
        >>> sorted([(fe.frame.name,fe.name) for fe in fn.fes('sound')]) # doctest: +NORMALIZE_WHITESPACE
        [('Cause_to_make_noise', 'Sound_maker'), ('Make_noise', 'Sound'),
         ('Make_noise', 'Sound_source'), ('Sound_movement', 'Location_of_sound_source'),
         ('Sound_movement', 'Sound'), ('Sound_movement', 'Sound_source'),
         ('Sounds', 'Component_sound'), ('Sounds', 'Location_of_sound_source'),
         ('Sounds', 'Sound_source'), ('Vocalizations', 'Location_of_sound_source'),
         ('Vocalizations', 'Sound_source')]
        >>> sorted([(fe.frame.name,fe.name) for fe in fn.fes('sound',r'(?i)make_noise')]) # doctest: +NORMALIZE_WHITESPACE
        [('Cause_to_make_noise', 'Sound_maker'),
         ('Make_noise', 'Sound'),
         ('Make_noise', 'Sound_source')]
        >>> sorted(set(fe.name for fe in fn.fes('^sound')))
        ['Sound', 'Sound_maker', 'Sound_source']
        >>> len(fn.fes('^sound$'))
        2

        :param name: A regular expression pattern used to match against
            frame element names. If 'name' is None, then a list of all
            frame elements will be returned.
        :type name: str
        :return: A list of matching frame elements
        :rtype: list(AttrDict)
        """
    if frame is not None:
        if isinstance(frame, int):
            frames = [self.frame(frame)]
        elif isinstance(frame, str):
            frames = self.frames(frame)
        else:
            frames = [frame]
    else:
        frames = self.frames()
    return PrettyList((fe for f in frames for fename, fe in f.FE.items() if name is None or re.search(name, fename, re.I)))