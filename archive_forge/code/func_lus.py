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
def lus(self, name=None, frame=None):
    """
        Obtain details for lexical units.
        Optionally restrict by lexical unit name pattern, and/or to a certain frame
        or frames whose name matches a pattern.

        >>> from nltk.corpus import framenet as fn
        >>> len(fn.lus()) in (11829, 13572) # FN 1.5 and 1.7, resp.
        True
        >>> PrettyList(sorted(fn.lus(r'(?i)a little'), key=itemgetter('ID')), maxReprSize=0, breakLines=True)
        [<lu ID=14733 name=a little.n>,
         <lu ID=14743 name=a little.adv>,
         <lu ID=14744 name=a little bit.adv>]
        >>> PrettyList(sorted(fn.lus(r'interest', r'(?i)stimulus'), key=itemgetter('ID')))
        [<lu ID=14894 name=interested.a>, <lu ID=14920 name=interesting.a>]

        A brief intro to Lexical Units (excerpted from "FrameNet II:
        Extended Theory and Practice" by Ruppenhofer et. al., 2010):

        A lexical unit (LU) is a pairing of a word with a meaning. For
        example, the "Apply_heat" Frame describes a common situation
        involving a Cook, some Food, and a Heating Instrument, and is
        _evoked_ by words such as bake, blanch, boil, broil, brown,
        simmer, steam, etc. These frame-evoking words are the LUs in the
        Apply_heat frame. Each sense of a polysemous word is a different
        LU.

        We have used the word "word" in talking about LUs. The reality
        is actually rather complex. When we say that the word "bake" is
        polysemous, we mean that the lemma "bake.v" (which has the
        word-forms "bake", "bakes", "baked", and "baking") is linked to
        three different frames:

           - Apply_heat: "Michelle baked the potatoes for 45 minutes."

           - Cooking_creation: "Michelle baked her mother a cake for her birthday."

           - Absorb_heat: "The potatoes have to bake for more than 30 minutes."

        These constitute three different LUs, with different
        definitions.

        Multiword expressions such as "given name" and hyphenated words
        like "shut-eye" can also be LUs. Idiomatic phrases such as
        "middle of nowhere" and "give the slip (to)" are also defined as
        LUs in the appropriate frames ("Isolated_places" and "Evading",
        respectively), and their internal structure is not analyzed.

        Framenet provides multiple annotated examples of each sense of a
        word (i.e. each LU).  Moreover, the set of examples
        (approximately 20 per LU) illustrates all of the combinatorial
        possibilities of the lexical unit.

        Each LU is linked to a Frame, and hence to the other words which
        evoke that Frame. This makes the FrameNet database similar to a
        thesaurus, grouping together semantically similar words.

        In the simplest case, frame-evoking words are verbs such as
        "fried" in:

           "Matilde fried the catfish in a heavy iron skillet."

        Sometimes event nouns may evoke a Frame. For example,
        "reduction" evokes "Cause_change_of_scalar_position" in:

           "...the reduction of debt levels to $665 million from $2.6 billion."

        Adjectives may also evoke a Frame. For example, "asleep" may
        evoke the "Sleep" frame as in:

           "They were asleep for hours."

        Many common nouns, such as artifacts like "hat" or "tower",
        typically serve as dependents rather than clearly evoking their
        own frames.

        :param name: A regular expression pattern used to search the LU
            names. Note that LU names take the form of a dotted
            string (e.g. "run.v" or "a little.adv") in which a
            lemma precedes the "." and a POS follows the
            dot. The lemma may be composed of a single lexeme
            (e.g. "run") or of multiple lexemes (e.g. "a
            little"). If 'name' is not given, then all LUs will
            be returned.

            The valid POSes are:

                   v    - verb
                   n    - noun
                   a    - adjective
                   adv  - adverb
                   prep - preposition
                   num  - numbers
                   intj - interjection
                   art  - article
                   c    - conjunction
                   scon - subordinating conjunction

        :type name: str
        :type frame: str or int or frame
        :return: A list of selected (or all) lexical units
        :rtype: list of LU objects (dicts). See the lu() function for info
          about the specifics of LU objects.

        """
    if not self._lu_idx:
        self._buildluindex()
    if name is not None:
        result = PrettyList((self.lu(luID) for luID, luName in self.lu_ids_and_names(name).items()))
        if frame is not None:
            if isinstance(frame, int):
                frameIDs = {frame}
            elif isinstance(frame, str):
                frameIDs = {f.ID for f in self.frames(frame)}
            else:
                frameIDs = {frame.ID}
            result = PrettyList((lu for lu in result if lu.frame.ID in frameIDs))
    elif frame is not None:
        if isinstance(frame, int):
            frames = [self.frame(frame)]
        elif isinstance(frame, str):
            frames = self.frames(frame)
        else:
            frames = [frame]
        result = PrettyLazyIteratorList(iter(LazyConcatenation((list(f.lexUnit.values()) for f in frames))))
    else:
        luIDs = [luID for luID, lu in self._lu_idx.items() if lu.status not in self._bad_statuses]
        result = PrettyLazyMap(self.lu, luIDs)
    return result