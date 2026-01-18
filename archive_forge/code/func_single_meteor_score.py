from itertools import chain, product
from typing import Callable, Iterable, List, Tuple
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
def single_meteor_score(reference: Iterable[str], hypothesis: Iterable[str], preprocess: Callable[[str], str]=str.lower, stemmer: StemmerI=PorterStemmer(), wordnet: WordNetCorpusReader=wordnet, alpha: float=0.9, beta: float=3.0, gamma: float=0.5) -> float:
    """
    Calculates METEOR score for single hypothesis and reference as per
    "Meteor: An Automatic Metric for MT Evaluation with HighLevels of
    Correlation with Human Judgments" by Alon Lavie and Abhaya Agarwal,
    in Proceedings of ACL.
    https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']

    >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']


    >>> round(single_meteor_score(reference1, hypothesis1),4)
    0.6944

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(single_meteor_score(['this', 'is', 'a', 'cat'], ['non', 'matching', 'hypothesis']),4)
    0.0

    :param reference: pre-tokenized reference
    :param hypothesis: pre-tokenized hypothesis
    :param preprocess: preprocessing function (default str.lower)
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :param alpha: parameter for controlling relative weights of precision and recall.
    :param beta: parameter for controlling shape of penalty as a
                 function of as a function of fragmentation.
    :param gamma: relative weight assigned to fragmentation penalty.
    :return: The sentence-level METEOR score.
    """
    enum_hypothesis, enum_reference = _generate_enums(hypothesis, reference, preprocess=preprocess)
    translation_length = len(enum_hypothesis)
    reference_length = len(enum_reference)
    matches, _, _ = _enum_align_words(enum_hypothesis, enum_reference, stemmer=stemmer, wordnet=wordnet)
    matches_count = len(matches)
    try:
        precision = float(matches_count) / translation_length
        recall = float(matches_count) / reference_length
        fmean = precision * recall / (alpha * precision + (1 - alpha) * recall)
        chunk_count = float(_count_chunks(matches))
        frag_frac = chunk_count / matches_count
    except ZeroDivisionError:
        return 0.0
    penalty = gamma * frag_frac ** beta
    return (1 - penalty) * fmean