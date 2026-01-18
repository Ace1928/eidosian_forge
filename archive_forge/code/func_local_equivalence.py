from __future__ import annotations
from math import sqrt
import numpy as np
def local_equivalence(weyl: np.ndarray) -> np.ndarray:
    """Computes the equivalent local invariants from the
    Weyl coordinates.

    Args:
        weyl (ndarray): Weyl coordinates.

    Returns:
        ndarray: Local equivalent coordinates [g0, g1, g3].

    Notes:
        This uses Eq. 30 from Zhang et al, PRA 67, 042313 (2003),
        but we multiply weyl coordinates by 2 since we are
        working in the reduced chamber.
    """
    g0_equiv = np.prod(np.cos(2 * weyl) ** 2) - np.prod(np.sin(2 * weyl) ** 2)
    g1_equiv = np.prod(np.sin(4 * weyl)) / 4
    g2_equiv = 4 * np.prod(np.cos(2 * weyl) ** 2) - 4 * np.prod(np.sin(2 * weyl) ** 2) - np.prod(np.cos(4 * weyl))
    return np.round([g0_equiv, g1_equiv, g2_equiv], 12) + 0.0