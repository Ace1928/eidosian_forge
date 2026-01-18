from nltk.tag.api import TaggerI
from nltk.tag.util import str2tuple, tuple2str, untag
from nltk.tag.sequential import (
from nltk.tag.brill import BrillTagger
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.tag.tnt import TnT
from nltk.tag.hunpos import HunposTagger
from nltk.tag.stanford import StanfordTagger, StanfordPOSTagger, StanfordNERTagger
from nltk.tag.hmm import HiddenMarkovModelTagger, HiddenMarkovModelTrainer
from nltk.tag.senna import SennaTagger, SennaChunkTagger, SennaNERTagger
from nltk.tag.mapping import tagset_mapping, map_tag
from nltk.tag.crf import CRFTagger
from nltk.tag.perceptron import PerceptronTagger
from nltk.data import load, find
def pos_tag_sents(sentences, tagset=None, lang='eng'):
    """
    Use NLTK's currently recommended part of speech tagger to tag the
    given list of sentences, each consisting of a list of tokens.

    :param sentences: List of sentences to be tagged
    :type sentences: list(list(str))
    :param tagset: the tagset to be used, e.g. universal, wsj, brown
    :type tagset: str
    :param lang: the ISO 639 code of the language, e.g. 'eng' for English, 'rus' for Russian
    :type lang: str
    :return: The list of tagged sentences
    :rtype: list(list(tuple(str, str)))
    """
    tagger = _get_tagger(lang)
    return [_pos_tag(sent, tagset, tagger, lang) for sent in sentences]