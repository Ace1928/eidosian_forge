import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_custom_scorer(self):
    """Test using a custom scoring function."""
    bigram = Phrases(self.sentences, min_count=1, threshold=0.001, scoring=dumb_scorer, connector_words=self.connector_words)
    test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
    seen_scores = list(bigram.find_phrases(test_sentences).values())
    assert all(seen_scores)
    assert len(seen_scores) == 2