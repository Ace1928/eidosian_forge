from .polynomial import Polynomial
from .fieldExtensions import my_rnfequation
from ..pari import pari
def item_to_pari(item):
    key, value = item
    if pari_number_field:
        return (key, pari(value).Mod(pari_number_field))
    else:
        return (key, pari(value))