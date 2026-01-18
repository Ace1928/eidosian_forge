from itertools import chain, product
from typing import Callable, Iterable, List, Tuple
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
def stem_match(hypothesis: Iterable[str], reference: Iterable[str], stemmer: StemmerI=PorterStemmer()) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between hypothesis and reference

    :param hypothesis: pre-tokenized hypothesis
    :param reference: pre-tokenized reference
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_stem_match(enum_hypothesis_list, enum_reference_list, stemmer=stemmer)