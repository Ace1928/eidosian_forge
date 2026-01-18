import contextlib
from pathlib import Path
from unittest import mock
import pytest
import cartopy
import cartopy.io as cio
from cartopy.io.shapereader import NEShpDownloader
@contextlib.contextmanager
def config_replace(replacement_dict):
    """
    Provides a context manager to replace the ``cartopy.config['downloaders']``
    dict with the given dictionary. Great for testing purposes!

    """
    with contextlib.ExitStack() as stack:
        stack.callback(cartopy.config.__setitem__, 'downloaders', cartopy.config['downloaders'])
        cartopy.config['downloaders'] = replacement_dict
        yield