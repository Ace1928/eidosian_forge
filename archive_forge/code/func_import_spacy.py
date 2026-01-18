import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union
def import_spacy() -> Any:
    """Import the spacy python package and raise an error if it is not installed."""
    try:
        import spacy
    except ImportError:
        raise ImportError('This callback manager requires the `spacy` python package installed. Please install it with `pip install spacy`')
    return spacy