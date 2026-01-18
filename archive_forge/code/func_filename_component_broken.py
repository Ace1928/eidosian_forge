import re
from pathlib import Path
from typing import Union
from .extern import packaging
def filename_component_broken(value: str) -> str:
    """
    Produce the incorrect filename component for compatibility.

    See pypa/setuptools#4167 for detailed analysis.

    TODO: replace this with filename_component after pip 24 is
    nearly-ubiquitous.

    >>> filename_component_broken('foo_bar-baz')
    'foo-bar-baz'
    """
    return value.replace('_', '-')