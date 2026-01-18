import logging
from . import Auth
from .AppAuthentication import AppAuthentication
from .GithubException import (
from .GithubIntegration import GithubIntegration
from .GithubRetry import GithubRetry
from .InputFileContent import InputFileContent
from .InputGitAuthor import InputGitAuthor
from .InputGitTreeElement import InputGitTreeElement
from .MainClass import Github
def enable_console_debug_logging() -> None:
    """
    This function sets up a very simple logging configuration (log everything on standard output) that is useful for troubleshooting.
    """
    set_log_level(logging.DEBUG)