from typing import Callable, Dict, List, Optional, Tuple
from thinc.api import Model
from ...pipeline import Lemmatizer
from ...pipeline.lemmatizer import lemmatizer_score
from ...symbols import POS
from ...tokens import Token
from ...vocab import Vocab
def pymorphy3_lookup_lemmatize(self, token: Token) -> List[str]:
    return self._pymorphy_lookup_lemmatize(token)