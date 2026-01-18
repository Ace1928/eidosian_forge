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
def exemplars(self, luNamePattern=None, frame=None, fe=None, fe2=None):
    """
        Lexicographic exemplar sentences, optionally filtered by LU name and/or 1-2 FEs that
        are realized overtly. 'frame' may be a name pattern, frame ID, or frame instance.
        'fe' may be a name pattern or FE instance; if specified, 'fe2' may also
        be specified to retrieve sentences with both overt FEs (in either order).
        """
    if fe is None and fe2 is not None:
        raise FramenetError('exemplars(..., fe=None, fe2=<value>) is not allowed')
    elif fe is not None and fe2 is not None:
        if not isinstance(fe2, str):
            if isinstance(fe, str):
                fe, fe2 = (fe2, fe)
            elif fe.frame is not fe2.frame:
                raise FramenetError('exemplars() call with inconsistent `fe` and `fe2` specification (frames must match)')
    if frame is None and fe is not None and (not isinstance(fe, str)):
        frame = fe.frame
    lusByFrame = defaultdict(list)
    if frame is not None or luNamePattern is not None:
        if frame is None or isinstance(frame, str):
            if luNamePattern is not None:
                frames = set()
                for lu in self.lus(luNamePattern, frame=frame):
                    frames.add(lu.frame.ID)
                    lusByFrame[lu.frame.name].append(lu)
                frames = LazyMap(self.frame, list(frames))
            else:
                frames = self.frames(frame)
        else:
            if isinstance(frame, int):
                frames = [self.frame(frame)]
            else:
                frames = [frame]
            if luNamePattern is not None:
                lusByFrame = {frame.name: self.lus(luNamePattern, frame=frame)}
        if fe is not None:
            if isinstance(fe, str):
                frames = PrettyLazyIteratorList((f for f in frames if fe in f.FE or any((re.search(fe, ffe, re.I) for ffe in f.FE.keys()))))
            else:
                if fe.frame not in frames:
                    raise FramenetError('exemplars() call with inconsistent `frame` and `fe` specification')
                frames = [fe.frame]
            if fe2 is not None:
                if isinstance(fe2, str):
                    frames = PrettyLazyIteratorList((f for f in frames if fe2 in f.FE or any((re.search(fe2, ffe, re.I) for ffe in f.FE.keys()))))
    elif fe is not None:
        frames = {ffe.frame.ID for ffe in self.fes(fe)}
        if fe2 is not None:
            frames2 = {ffe.frame.ID for ffe in self.fes(fe2)}
            frames = frames & frames2
        frames = LazyMap(self.frame, list(frames))
    else:
        frames = self.frames()

    def _matching_exs():
        for f in frames:
            fes = fes2 = None
            if fe is not None:
                fes = {ffe for ffe in f.FE.keys() if re.search(fe, ffe, re.I)} if isinstance(fe, str) else {fe.name}
                if fe2 is not None:
                    fes2 = {ffe for ffe in f.FE.keys() if re.search(fe2, ffe, re.I)} if isinstance(fe2, str) else {fe2.name}
            for lu in lusByFrame[f.name] if luNamePattern is not None else f.lexUnit.values():
                for ex in lu.exemplars:
                    if (fes is None or self._exemplar_of_fes(ex, fes)) and (fes2 is None or self._exemplar_of_fes(ex, fes2)):
                        yield ex
    return PrettyLazyIteratorList(_matching_exs())