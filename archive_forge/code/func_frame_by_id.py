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
def frame_by_id(self, fn_fid, ignorekeys=[]):
    """
        Get the details for the specified Frame using the frame's id
        number.

        Usage examples:

        >>> from nltk.corpus import framenet as fn
        >>> f = fn.frame_by_id(256)
        >>> f.ID
        256
        >>> f.name
        'Medical_specialties'
        >>> f.definition # doctest: +NORMALIZE_WHITESPACE
        "This frame includes words that name medical specialties and is closely related to the
        Medical_professionals frame.  The FE Type characterizing a sub-are in a Specialty may also be
        expressed. 'Ralph practices paediatric oncology.'"

        :param fn_fid: The Framenet id number of the frame
        :type fn_fid: int
        :param ignorekeys: The keys to ignore. These keys will not be
            included in the output. (optional)
        :type ignorekeys: list(str)
        :return: Information about a frame
        :rtype: dict

        Also see the ``frame()`` function for details about what is
        contained in the dict that is returned.
        """
    try:
        fentry = self._frame_idx[fn_fid]
        if '_type' in fentry:
            return fentry
        name = fentry['name']
    except TypeError:
        self._buildframeindex()
        name = self._frame_idx[fn_fid]['name']
    except KeyError as e:
        raise FramenetError(f'Unknown frame id: {fn_fid}') from e
    return self.frame_by_name(name, ignorekeys, check_cache=False)