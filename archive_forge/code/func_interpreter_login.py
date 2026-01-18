import os
import subprocess
from functools import partial
from getpass import getpass
from pathlib import Path
from typing import Optional
from . import constants
from .commands._cli_utils import ANSI
from .utils import (
from .utils._token import _get_token_from_environment, _get_token_from_google_colab
def interpreter_login(new_session: bool=True, write_permission: bool=False) -> None:
    """
    Displays a prompt to login to the HF website and store the token.

    This is equivalent to [`login`] without passing a token when not run in a notebook.
    [`interpreter_login`] is useful if you want to force the use of the terminal prompt
    instead of a notebook widget.

    For more details, see [`login`].

    Args:
        new_session (`bool`, defaults to `True`):
            If `True`, will request a token even if one is already saved on the machine.
        write_permission (`bool`, defaults to `False`):
            If `True`, requires a token with write permission.

    """
    if not new_session and _current_token_okay(write_permission=write_permission):
        print('User is already logged in.')
        return
    from .commands.delete_cache import _ask_for_confirmation_no_tui
    print(_HF_LOGO_ASCII)
    if get_token() is not None:
        print('    A token is already saved on your machine. Run `huggingface-cli whoami` to get more information or `huggingface-cli logout` if you want to log out.')
        print('    Setting a new token will erase the existing one.')
    print('    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .')
    if os.name == 'nt':
        print("Token can be pasted using 'Right-Click'.")
    token = getpass('Enter your token (input will not be visible): ')
    add_to_git_credential = _ask_for_confirmation_no_tui('Add token as git credential?')
    _login(token=token, add_to_git_credential=add_to_git_credential, write_permission=write_permission)