from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import RegexpTokenizer
def hyp_extra(self, toktype, debug=True):
    """
        Compute the extraneous material in the hypothesis.

        :param toktype: distinguish Named Entities from ordinary words
        :type toktype: 'ne' or 'word'
        """
    ne_extra = {token for token in self._hyp_extra if self._ne(token)}
    if toktype == 'ne':
        return ne_extra
    elif toktype == 'word':
        return self._hyp_extra - ne_extra
    else:
        raise ValueError("Type not recognized: '%s'" % toktype)