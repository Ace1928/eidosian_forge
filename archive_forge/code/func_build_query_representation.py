import math
from collections.abc import Sequence
import heapq
import json
import torch
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
def build_query_representation(self, query):
    """
        Build representation of query, e.g. words or n-grams.

        :param query: string to represent.

        :returns: dictionary containing 'words' dictionary (token => frequency)
                  and 'norm' float (square root of the number of tokens)
        """
    rep = {}
    rep['words'] = {}
    words = [w for w in self.dictionary.tokenize(query.lower())]
    rw = rep['words']
    used = {}
    for w in words:
        if len(self.dictionary.freq) > 0:
            rw[w] = 1.0 / (1.0 + math.log(1.0 + self.dictionary.freq[w]))
        elif w not in stopwords:
            rw[w] = 1
        used[w] = True
    rep['norm'] = math.sqrt(len(words))
    return rep