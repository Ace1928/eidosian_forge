import math
import secrets
import uuid as _uu
from typing import List
from typing import Optional
def set_alphabet(self, alphabet: str) -> None:
    """Set the alphabet to be used for new UUIDs."""
    new_alphabet = list(sorted(set(alphabet)))
    if len(new_alphabet) > 1:
        self._alphabet = new_alphabet
        self._alpha_len = len(self._alphabet)
    else:
        raise ValueError('Alphabet with more than one unique symbols required.')