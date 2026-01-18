from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def nltkdemo18plus():
    """
    Return 18 templates, from the original nltk demo, and additionally a few
    multi-feature ones (the motivation is easy comparison with nltkdemo18)
    """
    return nltkdemo18() + [Template(Word([-1]), Pos([1])), Template(Pos([-1]), Word([1])), Template(Word([-1]), Word([0]), Pos([1])), Template(Pos([-1]), Word([0]), Word([1])), Template(Pos([-1]), Word([0]), Pos([1]))]