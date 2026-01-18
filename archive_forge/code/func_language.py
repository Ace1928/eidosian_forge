from typing import List, Optional, Union
def language(self, language: str) -> 'Query':
    """
        Analyze the query as being in the specified language.

        :param language: The language (e.g. `chinese` or `english`)
        """
    self._language = language
    return self