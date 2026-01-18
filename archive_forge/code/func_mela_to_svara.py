import re
import numpy as np
from numba import jit
from .intervals import INTERVALS
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Dict, List, Union, overload
from ..util.decorators import vectorize
from .._typing import _ScalarOrSequence, _FloatLike_co, _SequenceLike
@cache(level=10)
def mela_to_svara(mela: Union[str, int], *, abbr: bool=True, unicode: bool=True) -> List[str]:
    """Spell the Carnatic svara names for a given melakarta raga

    This function exists to resolve enharmonic equivalences between
    pitch classes:

        - Ri2 / Ga1
        - Ri3 / Ga2
        - Dha2 / Ni1
        - Dha3 / Ni2

    For svara outside the raga, names are chosen to preserve orderings
    so that all Ri precede all Ga, and all Dha precede all Ni.

    Parameters
    ----------
    mela : str or int
        the name or numerical index of the melakarta raga

    abbr : bool
        If `True`, use single-letter svara names: S, R, G, ...

        If `False`, use full names: Sa, Ri, Ga, ...

    unicode : bool
        If `True`, use unicode symbols for numberings, e.g., Ri₁

        If `False`, use low-order ASCII, e.g., Ri1.

    Returns
    -------
    svara : list of strings

        The svara names for each of the 12 pitch classes.

    See Also
    --------
    key_to_notes
    mela_to_degrees
    list_mela

    Examples
    --------
    Melakarta #1 (Kanakangi) uses R1, G1, D1, N1

    >>> librosa.mela_to_svara(1)
    ['S', 'R₁', 'G₁', 'G₂', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'N₁', 'N₂', 'N₃']

    #19 (Jhankaradhwani) uses R2 and G2 so the third svara are Ri:

    >>> librosa.mela_to_svara(19)
    ['S', 'R₁', 'R₂', 'G₂', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'N₁', 'N₂', 'N₃']

    #31 (Yagapriya) uses R3 and G3, so third and fourth svara are Ri:

    >>> librosa.mela_to_svara(31)
    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'N₁', 'N₂', 'N₃']

    #34 (Vagadheeswari) uses D2 and N2, so Ni1 becomes Dha2:

    >>> librosa.mela_to_svara(34)
    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'D₂', 'N₂', 'N₃']

    #36 (Chalanatta) uses D3 and N3, so Ni2 becomes Dha3:

    >>> librosa.mela_to_svara(36)
    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'D₂', 'D₃', 'N₃']

    # You can also query by raga name instead of index:

    >>> librosa.mela_to_svara('chalanatta')
    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'D₂', 'D₃', 'N₃']
    """
    svara_map = ['Sa', 'Ri₁', '', '', 'Ga₃', 'Ma₁', 'Ma₂', 'Pa', 'Dha₁', '', '', 'Ni₃']
    if isinstance(mela, str):
        mela_idx = MELAKARTA_MAP[mela.lower()] - 1
    elif 0 < mela <= 72:
        mela_idx = mela - 1
    else:
        raise ParameterError(f'mela={mela} must be in range [1, 72]')
    lower = mela_idx % 36
    if lower < 6:
        svara_map[2] = 'Ga₁'
    else:
        svara_map[2] = 'Ri₂'
    if lower < 30:
        svara_map[3] = 'Ga₂'
    else:
        svara_map[3] = 'Ri₃'
    upper = mela_idx % 6
    if upper == 0:
        svara_map[9] = 'Ni₁'
    else:
        svara_map[9] = 'Dha₂'
    if upper == 5:
        svara_map[10] = 'Dha₃'
    else:
        svara_map[10] = 'Ni₂'
    if abbr:
        t_abbr = str.maketrans({'a': '', 'h': '', 'i': ''})
        svara_map = [s.translate(t_abbr) for s in svara_map]
    if not unicode:
        t_uni = str.maketrans({'₁': '1', '₂': '2', '₃': '3'})
        svara_map = [s.translate(t_uni) for s in svara_map]
    return list(svara_map)