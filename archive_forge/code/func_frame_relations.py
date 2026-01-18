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
def frame_relations(self, frame=None, frame2=None, type=None):
    """
        :param frame: (optional) frame object, name, or ID; only relations involving
            this frame will be returned
        :param frame2: (optional; 'frame' must be a different frame) only show relations
            between the two specified frames, in either direction
        :param type: (optional) frame relation type (name or object); show only relations
            of this type
        :type frame: int or str or AttrDict
        :return: A list of all of the frame relations in framenet
        :rtype: list(dict)

        >>> from nltk.corpus import framenet as fn
        >>> frels = fn.frame_relations()
        >>> isinstance(frels, list)
        True
        >>> len(frels) in (1676, 2070)  # FN 1.5 and 1.7, resp.
        True
        >>> PrettyList(fn.frame_relations('Cooking_creation'), maxReprSize=0, breakLines=True)
        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>,
         <Parent=Apply_heat -- Using -> Child=Cooking_creation>,
         <MainEntry=Apply_heat -- See_also -> ReferringEntry=Cooking_creation>]
        >>> PrettyList(fn.frame_relations(274), breakLines=True)
        [<Parent=Avoiding -- Inheritance -> Child=Dodging>,
         <Parent=Avoiding -- Inheritance -> Child=Evading>, ...]
        >>> PrettyList(fn.frame_relations(fn.frame('Cooking_creation')), breakLines=True)
        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>,
         <Parent=Apply_heat -- Using -> Child=Cooking_creation>, ...]
        >>> PrettyList(fn.frame_relations('Cooking_creation', type='Inheritance'))
        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>]
        >>> PrettyList(fn.frame_relations('Cooking_creation', 'Apply_heat'), breakLines=True) # doctest: +NORMALIZE_WHITESPACE
        [<Parent=Apply_heat -- Using -> Child=Cooking_creation>,
        <MainEntry=Apply_heat -- See_also -> ReferringEntry=Cooking_creation>]
        """
    relation_type = type
    if not self._frel_idx:
        self._buildrelationindex()
    rels = None
    if relation_type is not None:
        if not isinstance(relation_type, dict):
            type = [rt for rt in self.frame_relation_types() if rt.name == type][0]
            assert isinstance(type, dict)
    if frame is not None:
        if isinstance(frame, dict) and 'frameRelations' in frame:
            rels = PrettyList(frame.frameRelations)
        else:
            if not isinstance(frame, int):
                if isinstance(frame, dict):
                    frame = frame.ID
                else:
                    frame = self.frame_by_name(frame).ID
            rels = [self._frel_idx[frelID] for frelID in self._frel_f_idx[frame]]
        if type is not None:
            rels = [rel for rel in rels if rel.type is type]
    elif type is not None:
        rels = type.frameRelations
    else:
        rels = self._frel_idx.values()
    if frame2 is not None:
        if frame is None:
            raise FramenetError('frame_relations(frame=None, frame2=<value>) is not allowed')
        if not isinstance(frame2, int):
            if isinstance(frame2, dict):
                frame2 = frame2.ID
            else:
                frame2 = self.frame_by_name(frame2).ID
        if frame == frame2:
            raise FramenetError('The two frame arguments to frame_relations() must be different frames')
        rels = [rel for rel in rels if rel.superFrame.ID == frame2 or rel.subFrame.ID == frame2]
    return PrettyList(sorted(rels, key=lambda frel: (frel.type.ID, frel.superFrameName, frel.subFrameName)))