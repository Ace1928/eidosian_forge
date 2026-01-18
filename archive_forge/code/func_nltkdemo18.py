from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def nltkdemo18():
    """
    Return 18 templates, from the original nltk demo, in multi-feature syntax
    """
    return [Template(Pos([-1])), Template(Pos([1])), Template(Pos([-2])), Template(Pos([2])), Template(Pos([-2, -1])), Template(Pos([1, 2])), Template(Pos([-3, -2, -1])), Template(Pos([1, 2, 3])), Template(Pos([-1]), Pos([1])), Template(Word([-1])), Template(Word([1])), Template(Word([-2])), Template(Word([2])), Template(Word([-2, -1])), Template(Word([1, 2])), Template(Word([-3, -2, -1])), Template(Word([1, 2, 3])), Template(Word([-1]), Word([1]))]