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
def notebook_login(new_session: bool=True, write_permission: bool=False) -> None:
    """
    Displays a widget to login to the HF website and store the token.

    This is equivalent to [`login`] without passing a token when run in a notebook.
    [`notebook_login`] is useful if you want to force the use of the notebook widget
    instead of a prompt in the terminal.

    For more details, see [`login`].

    Args:
        new_session (`bool`, defaults to `True`):
            If `True`, will request a token even if one is already saved on the machine.
        write_permission (`bool`, defaults to `False`):
            If `True`, requires a token with write permission.
    """
    try:
        import ipywidgets.widgets as widgets
        from IPython.display import display
    except ImportError:
        raise ImportError('The `notebook_login` function can only be used in a notebook (Jupyter or Colab) and you need the `ipywidgets` module: `pip install ipywidgets`.')
    if not new_session and _current_token_okay(write_permission=write_permission):
        print('User is already logged in.')
        return
    box_layout = widgets.Layout(display='flex', flex_flow='column', align_items='center', width='50%')
    token_widget = widgets.Password(description='Token:')
    git_checkbox_widget = widgets.Checkbox(value=True, description='Add token as git credential?')
    token_finish_button = widgets.Button(description='Login')
    login_token_widget = widgets.VBox([widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_START), token_widget, git_checkbox_widget, token_finish_button, widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_END)], layout=box_layout)
    display(login_token_widget)

    def login_token_event(t, write_permission: bool=False):
        """
        Event handler for the login button.

        Args:
            write_permission (`bool`, defaults to `False`):
                If `True`, requires a token with write permission.
        """
        token = token_widget.value
        add_to_git_credential = git_checkbox_widget.value
        token_widget.value = ''
        login_token_widget.children = [widgets.Label('Connecting...')]
        try:
            with capture_output() as captured:
                _login(token, add_to_git_credential=add_to_git_credential, write_permission=write_permission)
            message = captured.getvalue()
        except Exception as error:
            message = str(error)
        login_token_widget.children = [widgets.Label(line) for line in message.split('\n') if line.strip()]
    token_finish_button.on_click(partial(login_token_event, write_permission=write_permission))