import contextlib
import functools
import pluggy
import keyring.errors
@hookimpl()
@restore_signature
@suppress(keyring.errors.KeyringError)
def devpiclient_get_password(url, username):
    """
    >>> pluggy._hooks.varnames(devpiclient_get_password)
    (('url', 'username'), ())
    >>>
    """
    return keyring.get_password(url, username)