import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union
import requests
from outlines import generate, models
def download_from_github(short_path: str):
    """Download the file in which the function is stored on GitHub."""
    GITHUB_BASE_URL = 'https://raw.githubusercontent.com'
    BRANCH = 'main'
    path = short_path.split('/')
    if len(path) < 3:
        raise ValueError('Please provide a valid path in the form {USERNAME}/{REPO_NAME}/{PATH_TO_FILE}.')
    elif short_path[-3:] == '.py':
        raise ValueError('Do not append the `.py` extension to the program name.')
    username = path[0]
    repo = path[1]
    path_to_file = path[2:]
    url = '/'.join([GITHUB_BASE_URL, username, repo, BRANCH] + path_to_file) + '.py'
    result = requests.get(url)
    if result.status_code == 200:
        return result.text
    elif result.status_code == 404:
        raise ValueError(f'Program could not be found at {url}. Please make sure you entered the GitHub username, repository name and path to the program correctly.')
    else:
        result.raise_for_status()