import re
import subprocess
from typing import List, Optional
from ..constants import ENDPOINT
from ._subprocess import run_interactive_subprocess, run_subprocess
def set_git_credential(token: str, username: str='hf_user', folder: Optional[str]=None) -> None:
    """Save a username/token pair in git credential for HF Hub registry.

    Credentials are saved in all configured helpers (store, cache, macOS keychain,...).
    Calls "`git credential approve`" internally. See https://git-scm.com/docs/git-credential.

    Args:
        username (`str`, defaults to `"hf_user"`):
            A git username. Defaults to `"hf_user"`, the default user used in the Hub.
        token (`str`, defaults to `"hf_user"`):
            A git password. In practice, the User Access Token for the Hub.
            See https://huggingface.co/settings/tokens.
        folder (`str`, *optional*):
            The folder in which to check the configured helpers.
    """
    with run_interactive_subprocess('git credential approve', folder=folder) as (stdin, _):
        stdin.write(f'url={ENDPOINT}\nusername={username.lower()}\npassword={token}\n\n')
        stdin.flush()