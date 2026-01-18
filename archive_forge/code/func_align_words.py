from itertools import chain, product
from typing import Callable, Iterable, List, Tuple
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
def align_words(hypothesis: Iterable[str], reference: Iterable[str], stemmer: StemmerI=PorterStemmer(), wordnet: WordNetCorpusReader=wordnet) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    In case there are multiple matches the match which has the least number
    of crossing is chosen.

    :param hypothesis: pre-tokenized hypothesis
    :param reference: pre-tokenized reference
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :return: sorted list of matched tuples, unmatched hypothesis list, unmatched reference list
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_align_words(enum_hypothesis_list, enum_reference_list, stemmer=stemmer, wordnet=wordnet)