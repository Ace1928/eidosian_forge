from typing import List, Optional, Union
def scorer(self, scorer: str) -> 'Query':
    """
        Use a different scoring function to evaluate document relevance.
        Default is `TFIDF`.

        :param scorer: The scoring function to use
                       (e.g. `TFIDF.DOCNORM` or `BM25`)
        """
    self._scorer = scorer
    return self