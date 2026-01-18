import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def pprint_themroles(self, vnclass, indent=''):
    """Returns pretty printed version of thematic roles in a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet class's thematic roles.

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        """
    if isinstance(vnclass, str):
        vnclass = self.vnclass(vnclass)
    pieces = []
    for themrole in self.themroles(vnclass):
        piece = indent + '* ' + themrole.get('type')
        modifiers = [modifier['value'] + modifier['type'] for modifier in themrole['modifiers']]
        if modifiers:
            piece += '[{}]'.format(' '.join(modifiers))
        pieces.append(piece)
    return '\n'.join(pieces)