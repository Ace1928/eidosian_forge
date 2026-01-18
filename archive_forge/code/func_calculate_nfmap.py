import os
import tempfile
from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.classify.megam import call_megam, parse_megam_weights, write_megam_file
from nltk.classify.tadm import call_tadm, parse_tadm_weights, write_tadm_file
from nltk.classify.util import CutoffChecker, accuracy, log_likelihood
from nltk.data import gzip_open_unicode
from nltk.probability import DictionaryProbDist
from nltk.util import OrderedDict
def calculate_nfmap(train_toks, encoding):
    """
    Construct a map that can be used to compress ``nf`` (which is
    typically sparse).

    *nf(feature_vector)* is the sum of the feature values for
    *feature_vector*.

    This represents the number of features that are active for a
    given labeled text.  This method finds all values of *nf(t)*
    that are attested for at least one token in the given list of
    training tokens; and constructs a dictionary mapping these
    attested values to a continuous range *0...N*.  For example,
    if the only values of *nf()* that were attested were 3, 5, and
    7, then ``_nfmap`` might return the dictionary ``{3:0, 5:1, 7:2}``.

    :return: A map that can be used to compress ``nf`` to a dense
        vector.
    :rtype: dict(int -> int)
    """
    nfset = set()
    for tok, _ in train_toks:
        for label in encoding.labels():
            nfset.add(sum((val for id, val in encoding.encode(tok, label))))
    return {nf: i for i, nf in enumerate(nfset)}