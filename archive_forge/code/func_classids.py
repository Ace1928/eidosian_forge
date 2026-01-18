import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def classids(self, lemma=None, wordnetid=None, fileid=None, classid=None):
    """
        Return a list of the VerbNet class identifiers.  If a file
        identifier is specified, then return only the VerbNet class
        identifiers for classes (and subclasses) defined by that file.
        If a lemma is specified, then return only VerbNet class
        identifiers for classes that contain that lemma as a member.
        If a wordnetid is specified, then return only identifiers for
        classes that contain that wordnetid as a member.  If a classid
        is specified, then return only identifiers for subclasses of
        the specified VerbNet class.
        If nothing is specified, return all classids within VerbNet
        """
    if fileid is not None:
        return [c for c, f in self._class_to_fileid.items() if f == fileid]
    elif lemma is not None:
        return self._lemma_to_class[lemma]
    elif wordnetid is not None:
        return self._wordnet_to_class[wordnetid]
    elif classid is not None:
        xmltree = self.vnclass(classid)
        return [subclass.get('ID') for subclass in xmltree.findall('SUBCLASSES/VNSUBCLASS')]
    else:
        return sorted(self._class_to_fileid.keys())