import os
import stat
import sys
import textwrap
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union
from urllib.parse import urlparse
import click
import requests.utils
import wandb
from wandb.apis import InternalApi
from wandb.errors import term
from wandb.util import _is_databricks, isatty, prompt_choices
from .wburls import wburls
def prompt_api_key(settings: 'Settings', api: Optional[InternalApi]=None, input_callback: Optional[Callable]=None, browser_callback: Optional[Callable]=None, no_offline: bool=False, no_create: bool=False, local: bool=False) -> Union[str, bool, None]:
    """Prompt for api key.

    Returns:
        str - if key is configured
        None - if dryrun is selected
        False - if unconfigured (notty)
    """
    input_callback = input_callback or getpass
    log_string = term.LOG_STRING
    api = api or InternalApi(settings)
    anon_mode = _fixup_anon_mode(settings.anonymous)
    jupyter = settings._jupyter or False
    app_url = api.app_url
    choices = [choice for choice in LOGIN_CHOICES]
    if anon_mode == 'never':
        choices.remove(LOGIN_CHOICE_ANON)
    if jupyter and (not settings.login_timeout) or no_offline:
        choices.remove(LOGIN_CHOICE_DRYRUN)
    if jupyter and (not settings.login_timeout) or no_create:
        choices.remove(LOGIN_CHOICE_NEW)
    if jupyter and 'google.colab' in sys.modules:
        log_string = term.LOG_STRING_NOCOLOR
        key = wandb.jupyter.attempt_colab_login(app_url)
        if key is not None:
            write_key(settings, key, api=api)
            return key
    if anon_mode == 'must':
        result = LOGIN_CHOICE_ANON
    elif not jupyter and (not isatty(sys.stdout) or not isatty(sys.stdin)) or _is_databricks():
        result = LOGIN_CHOICE_NOTTY
    elif local:
        result = LOGIN_CHOICE_EXISTS
    elif len(choices) == 1:
        result = choices[0]
    else:
        result = prompt_choices(choices, input_timeout=settings.login_timeout, jupyter=jupyter)
    api_ask = f'{log_string}: Paste an API key from your profile and hit enter, or press ctrl+c to quit'
    if result == LOGIN_CHOICE_ANON:
        key = api.create_anonymous_api_key()
        write_key(settings, key, api=api, anonymous=True)
        return key
    elif result == LOGIN_CHOICE_NEW:
        key = browser_callback(signup=True) if browser_callback else None
        if not key:
            wandb.termlog(f'Create an account here: {app_url}/authorize?signup=true')
            key = input_callback(api_ask).strip()
        write_key(settings, key, api=api)
        return key
    elif result == LOGIN_CHOICE_EXISTS:
        key = browser_callback() if browser_callback else None
        if not key:
            if not (settings.is_local or local):
                host = app_url
                for prefix in ('http://', 'https://'):
                    if app_url.startswith(prefix):
                        host = app_url[len(prefix):]
                wandb.termlog(f'Logging into {host}. (Learn how to deploy a W&B server locally: {wburls.get('wandb_server')})')
            wandb.termlog(f'You can find your API key in your browser here: {app_url}/authorize')
            key = input_callback(api_ask).strip()
        write_key(settings, key, api=api)
        return key
    elif result == LOGIN_CHOICE_NOTTY:
        return False
    elif result == LOGIN_CHOICE_DRYRUN:
        return None
    else:
        key, anonymous = browser_callback() if jupyter and browser_callback else (None, False)
        write_key(settings, key, api=api)
        return key