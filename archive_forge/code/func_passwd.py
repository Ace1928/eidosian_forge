import getpass
import hashlib
import json
import os
import random
import traceback
import warnings
from contextlib import contextmanager
from jupyter_core.paths import jupyter_config_dir
from traitlets.config import Config
from traitlets.config.loader import ConfigFileNotFound, JSONFileConfigLoader
def passwd(passphrase=None, algorithm='argon2'):
    """Generate hashed password and salt for use in server configuration.

    In the server configuration, set `c.ServerApp.password` to
    the generated string.

    Parameters
    ----------
    passphrase : str
        Password to hash.  If unspecified, the user is asked to input
        and verify a password.
    algorithm : str
        Hashing algorithm to use (e.g, 'sha1' or any argument supported
        by :func:`hashlib.new`, or 'argon2').

    Returns
    -------
    hashed_passphrase : str
        Hashed password, in the format 'hash_algorithm:salt:passphrase_hash'.

    Examples
    --------
    >>> passwd("mypassword")  # doctest: +ELLIPSIS
    'argon2:...'

    """
    if passphrase is None:
        for _ in range(3):
            p0 = getpass.getpass('Enter password: ')
            p1 = getpass.getpass('Verify password: ')
            if p0 == p1:
                passphrase = p0
                break
            warnings.warn('Passwords do not match.', stacklevel=2)
        else:
            msg = 'No matching passwords found. Giving up.'
            raise ValueError(msg)
    if algorithm == 'argon2':
        import argon2
        ph = argon2.PasswordHasher(memory_cost=10240, time_cost=10, parallelism=8)
        h_ph = ph.hash(passphrase)
        return ':'.join((algorithm, h_ph))
    h = hashlib.new(algorithm)
    salt = ('%0' + str(salt_len) + 'x') % random.getrandbits(4 * salt_len)
    h.update(passphrase.encode('utf-8') + salt.encode('ascii'))
    return ':'.join((algorithm, salt, h.hexdigest()))