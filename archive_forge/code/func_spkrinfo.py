import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def spkrinfo(self, speaker):
    """
        :return: A dictionary mapping .. something.
        """
    if speaker in self._utterances:
        speaker = self.spkrid(speaker)
    if self._speakerinfo is None:
        self._speakerinfo = {}
        with self.open('spkrinfo.txt') as fp:
            for line in fp:
                if not line.strip() or line[0] == ';':
                    continue
                rec = line.strip().split(None, 9)
                key = f'dr{rec[2]}-{rec[1].lower()}{rec[0].lower()}'
                self._speakerinfo[key] = SpeakerInfo(*rec)
    return self._speakerinfo[speaker]