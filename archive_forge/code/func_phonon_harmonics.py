import numpy as np
from ase.parallel import world
from ase import units
from ase.md.md import process_temperature
def phonon_harmonics(force_constants, masses, temp=None, *, temperature_K=None, rng=np.random.rand, quantum=False, plus_minus=False, return_eigensolution=False, failfast=True):
    """Return displacements and velocities that produce a given temperature.

    Parameters:

    force_constants: array of size 3N x 3N
        force constants (Hessian) of the system in eV/Å²

    masses: array of length N
        masses of the structure in amu

    temp: float (deprecated)
        Temperature converted to eV (T * units.kB).  Deprecated, use 
        ``temperature_K``.

    temperature_K: float
        Temperature in Kelvin.

    rng: function
        Random number generator function, e.g., np.random.rand

    quantum: bool
        True for Bose-Einstein distribution, False for Maxwell-Boltzmann
        (classical limit)

    plus_minus: bool
        Displace atoms with +/- the amplitude accoding to PRB 94, 075125

    return_eigensolution: bool
        return eigenvalues and eigenvectors of the dynamical matrix

    failfast: bool
        True for sanity checking the phonon spectrum for negative
        frequencies at Gamma

    Returns:

    Displacements, velocities generated from the eigenmodes,
    (optional: eigenvalues, eigenvectors of dynamical matrix)

    Purpose:

    Excite phonon modes to specified temperature.

    This excites all phonon modes randomly so that each contributes,
    on average, equally to the given temperature.  Both potential
    energy and kinetic energy will be consistent with the phononic
    vibrations characteristic of the specified temperature.

    In other words the system will be equilibrated for an MD run at
    that temperature.

    force_constants should be the matrix as force constants, e.g.,
    as computed by the ase.phonons module.

    Let X_ai be the phonon modes indexed by atom and mode, w_i the
    phonon frequencies, and let 0 < Q_i <= 1 and 0 <= R_i < 1 be
    uniformly random numbers.  Then

    .. code-block:: none


                    1/2
       _     / k T \\     ---  1  _             1/2
       R  += | --- |      >  --- X   (-2 ln Q )    cos (2 pi R )
        a    \\  m  /     ---  w   ai         i                i
                 a        i    i


                    1/2
       _     / k T \\     --- _            1/2
       v   = | --- |      >  X  (-2 ln Q )    sin (2 pi R )
        a    \\  m  /     ---  ai        i                i
                 a        i

    Reference: [West, Estreicher; PRL 96, 22 (2006)]
    """
    temp = units.kB * process_temperature(temp, temperature_K, 'eV')
    rminv = (masses ** (-0.5)).repeat(3)
    dynamical_matrix = force_constants * rminv[:, None] * rminv[None, :]
    w2_s, X_is = np.linalg.eigh(dynamical_matrix)
    if failfast:
        zeros = w2_s[:3]
        worst_zero = np.abs(zeros).max()
        if worst_zero > 0.001:
            msg = 'Translational deviate from 0 significantly: '
            raise ValueError(msg + '{}'.format(w2_s[:3]))
        w2min = w2_s[3:].min()
        if w2min < 0:
            msg = 'Dynamical matrix has negative eigenvalues such as '
            raise ValueError(msg + '{}'.format(w2min))
    nw = len(w2_s) - 3
    n_atoms = len(masses)
    w_s = np.sqrt(w2_s[3:])
    X_acs = X_is[:, 3:].reshape(n_atoms, 3, nw)
    if quantum:
        hbar = units._hbar * units.J * units.s
        A_s = np.sqrt(hbar * (2 * n_BE(temp, hbar * w_s) + 1) / (2 * w_s))
    else:
        A_s = np.sqrt(temp) / w_s
    if plus_minus:
        spread = (-1) ** np.arange(nw)
        for ii in range(X_acs.shape[-1]):
            vec = X_acs[:, :, ii]
            max_arg = np.argmax(abs(vec))
            X_acs[:, :, ii] *= np.sign(vec.flat[max_arg])
        A_s *= spread
        phi_s = 2.0 * np.pi * rng(nw)
        v_ac = (w_s * A_s * np.sqrt(2) * np.cos(phi_s) * X_acs).sum(axis=2)
        v_ac /= np.sqrt(masses)[:, None]
        d_ac = (A_s * X_acs).sum(axis=2)
        d_ac /= np.sqrt(masses)[:, None]
    else:
        spread = np.sqrt(-2.0 * np.log(1.0 - rng(nw)))
        A_s *= spread
        phi_s = 2.0 * np.pi * rng(nw)
        v_ac = (w_s * A_s * np.cos(phi_s) * X_acs).sum(axis=2)
        v_ac /= np.sqrt(masses)[:, None]
        d_ac = (A_s * np.sin(phi_s) * X_acs).sum(axis=2)
        d_ac /= np.sqrt(masses)[:, None]
    if return_eigensolution:
        return (d_ac, v_ac, w2_s, X_is)
    return (d_ac, v_ac)