import enum
import os
import sys
from typing import Dict, Optional, Tuple
import click
import wandb
from wandb.errors import AuthenticationError, UsageError
from wandb.old.settings import Settings as OldSettings
from ..apis import InternalApi
from .internal.internal_api import Api
from .lib import apikey
from .wandb_settings import Settings, Source
def login_display(self):
    username = self._wl._get_username()
    if username:
        entity = self._entity or self._wl._get_entity()
        entity_str = ''
        if entity and entity in self._wl._get_teams() and (entity != username):
            entity_str = f' ({click.style(entity, fg='yellow')})'
        login_state_str = f'Currently logged in as: {click.style(username, fg='yellow')}{entity_str}'
    else:
        login_state_str = 'W&B API key is configured'
    login_info_str = f'Use {click.style('`wandb login --relogin`', bold=True)} to force relogin'
    wandb.termlog(f'{login_state_str}. {login_info_str}', repeat=False)