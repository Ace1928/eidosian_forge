import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def pprint_subclasses(self, vnclass, indent=''):
    """Returns pretty printed version of subclasses of VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet class's subclasses.

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        """
    if isinstance(vnclass, str):
        vnclass = self.vnclass(vnclass)
    subclasses = self.subclasses(vnclass)
    if not subclasses:
        subclasses = ['(none)']
    s = 'Subclasses: ' + ' '.join(subclasses)
    return textwrap.fill(s, 70, initial_indent=indent, subsequent_indent=indent + '  ')