import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def pprint_members(self, vnclass, indent=''):
    """Returns pretty printed version of members in a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet class's member verbs.

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        """
    if isinstance(vnclass, str):
        vnclass = self.vnclass(vnclass)
    members = self.lemmas(vnclass)
    if not members:
        members = ['(none)']
    s = 'Members: ' + ' '.join(members)
    return textwrap.fill(s, 70, initial_indent=indent, subsequent_indent=indent + '  ')