import functools
import warnings
import numpy as np
from ase.utils import IOContext
def bandgap(calc=None, direct=False, spin=None, output='-', eigenvalues=None, efermi=None, kpts=None):
    """Calculates the band-gap.

    Parameters:

    calc: Calculator object
        Electronic structure calculator object.
    direct: bool
        Calculate direct band-gap.
    spin: int or None
        For spin-polarized systems, you can use spin=0 or spin=1 to look only
        at a single spin-channel.
    output: file descriptor
        Use output=None for no text output or '-' for stdout (default).
    eigenvalues: ndarray of shape (nspin, nkpt, nband) or (nkpt, nband)
        Eigenvalues.
    efermi: float
        Fermi level (defaults to 0.0).
    kpts: ndarray of shape (nkpt, 3)
        For pretty text output only.

    Returns a (gap, p1, p2) tuple where p1 and p2 are tuples of indices of the
    valence and conduction points (s, k, n).

    Example:

    >>> gap, p1, p2 = bandgap(silicon.calc)
    Gap: 1.2 eV
    Transition (v -> c):
        [0.000, 0.000, 0.000] -> [0.500, 0.500, 0.000]
    >>> print(gap, p1, p2)
    1.2 (0, 0, 3), (0, 5, 4)
    >>> gap, p1, p2 = bandgap(silicon.calc, direct=True)
    Direct gap: 3.4 eV
    Transition at: [0.000, 0.000, 0.000]
    >>> print(gap, p1, p2)
    3.4 (0, 0, 3), (0, 0, 4)
    """
    if calc:
        kpts = calc.get_ibz_k_points()
        nk = len(kpts)
        ns = calc.get_number_of_spins()
        eigenvalues = np.array([[calc.get_eigenvalues(kpt=k, spin=s) for k in range(nk)] for s in range(ns)])
        if efermi is None:
            efermi = calc.get_fermi_level()
    efermi = efermi or 0.0
    e_skn = eigenvalues - efermi
    if eigenvalues.ndim == 2:
        e_skn = e_skn[np.newaxis]
    if not np.isfinite(e_skn).all():
        raise ValueError('Bad eigenvalues!')
    gap, (s1, k1, n1), (s2, k2, n2) = _bandgap(e_skn, spin, direct)
    with IOContext() as iocontext:
        fd = iocontext.openfile(output)
        p = functools.partial(print, file=fd)

        def skn(s, k, n):
            """Convert k or (s, k) to string."""
            if kpts is None:
                return '(s={}, k={}, n={})'.format(s, k, n)
            return '(s={}, k={}, n={}, [{:.2f}, {:.2f}, {:.2f}])'.format(s, k, n, *kpts[k])
        if spin is not None:
            p('spin={}: '.format(spin), end='')
        if gap == 0.0:
            p('No gap')
        elif direct:
            p('Direct gap: {:.3f} eV'.format(gap))
            if s1 == s2:
                p('Transition at:', skn(s1, k1, n1))
            else:
                p('Transition at:', skn('{}->{}'.format(s1, s2), k1, n1))
        else:
            p('Gap: {:.3f} eV'.format(gap))
            p('Transition (v -> c):')
            p(' ', skn(s1, k1, n1), '->', skn(s2, k2, n2))
    if eigenvalues.ndim != 3:
        p1 = (k1, n1)
        p2 = (k2, n2)
    else:
        p1 = (s1, k1, n1)
        p2 = (s2, k2, n2)
    return (gap, p1, p2)