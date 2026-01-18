import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def pprint_frames(self, vnclass, indent=''):
    """Returns pretty version of all frames in a VerbNet class

        Return a string containing a pretty-printed representation of
        the list of frames within the VerbNet class.

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        """
    if isinstance(vnclass, str):
        vnclass = self.vnclass(vnclass)
    pieces = []
    for vnframe in self.frames(vnclass):
        pieces.append(self._pprint_single_frame(vnframe, indent))
    return '\n'.join(pieces)