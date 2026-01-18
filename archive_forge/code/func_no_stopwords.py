from typing import List, Optional, Union
def no_stopwords(self) -> 'Query':
    """
        Prevent the query from being filtered for stopwords.
        Only useful in very big queries that you are certain contain
        no stopwords.
        """
    self._no_stopwords = True
    return self