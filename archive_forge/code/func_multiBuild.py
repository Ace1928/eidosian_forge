from reportlab.platypus.flowables import *
from reportlab.platypus.flowables import _ContainerSpace
from reportlab.lib.units import inch
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.frames import Frame
from reportlab.rl_config import defaultPageSize, verbose
import reportlab.lib.sequencer
from reportlab.pdfgen import canvas
from reportlab.lib.utils import isSeq, encode_label, decode_label, annotateException, strTypes
import sys
import logging
def multiBuild(self, story, maxPasses=10, **buildKwds):
    """Makes multiple passes until all indexing flowables
        are happy.

        Returns number of passes"""
    self._indexingFlowables = []
    for thing in story:
        if thing.isIndexing():
            self._indexingFlowables.append(thing)
    self._doSave = 0
    passes = 0
    mbe = []
    self._multiBuildEdits = mbe.append
    while 1:
        passes += 1
        if self._onProgress:
            self._onProgress('PASS', passes)
        if verbose:
            sys.stdout.write('building pass ' + str(passes) + '...')
        for fl in self._indexingFlowables:
            fl.beforeBuild()
        tempStory = story[:]
        self.build(tempStory, **buildKwds)
        for fl in self._indexingFlowables:
            fl.afterBuild()
        happy = self._allSatisfied()
        if happy:
            self._doSave = 0
            self.canv.save()
            break
        if passes > maxPasses:
            raise IndexError('Index entries not resolved after %d passes' % maxPasses)
        while mbe:
            e = mbe.pop(0)
            e[0](*e[1:])
    del self._multiBuildEdits
    if verbose:
        print('saved')
    return passes