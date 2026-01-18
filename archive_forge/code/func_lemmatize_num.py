import re
from typing import List, Optional, Tuple
from ...pipeline import Lemmatizer
from ...tokens import Token
def lemmatize_num(self, word: str, features: List[str], rule: str, index: List[str]) -> List[str]:
    """
        Lemmatize a numeral.

        word (str): The word to lemmatize.
        features (List[str]): The morphological features as a list of Feat=Val
            pairs.
        index (List[str]): The POS-specific lookup list.

        RETURNS (List[str]): The list of lemmas.
        """
    for old, new in self.lookups.get_table('lemma_rules').get('num', []):
        if word == old:
            return [new]
    splitted_word = word.split(',')
    if re.search('(\\.)([0-9]{3})$', splitted_word[0]):
        word = re.sub('\\.', '', word)
    word = re.sub(',', '.', word)
    return [word]