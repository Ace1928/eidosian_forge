from math import log
from operator import itemgetter
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tag.api import TaggerI
def basic_sent_chop(data, raw=True):
    """
    Basic method for tokenizing input into sentences
    for this tagger:

    :param data: list of tokens (words or (word, tag) tuples)
    :type data: str or tuple(str, str)
    :param raw: boolean flag marking the input data
                as a list of words or a list of tagged words
    :type raw: bool
    :return: list of sentences
             sentences are a list of tokens
             tokens are the same as the input

    Function takes a list of tokens and separates the tokens into lists
    where each list represents a sentence fragment
    This function can separate both tagged and raw sequences into
    basic sentences.

    Sentence markers are the set of [,.!?]

    This is a simple method which enhances the performance of the TnT
    tagger. Better sentence tokenization will further enhance the results.
    """
    new_data = []
    curr_sent = []
    sent_mark = [',', '.', '?', '!']
    if raw:
        for word in data:
            if word in sent_mark:
                curr_sent.append(word)
                new_data.append(curr_sent)
                curr_sent = []
            else:
                curr_sent.append(word)
    else:
        for word, tag in data:
            if word in sent_mark:
                curr_sent.append((word, tag))
                new_data.append(curr_sent)
                curr_sent = []
            else:
                curr_sent.append((word, tag))
    return new_data