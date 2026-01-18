import warnings
from typing import TYPE_CHECKING, Optional
def get_verbose() -> bool:
    """Get the value of the `verbose` global setting."""
    import langchain
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Importing verbose from langchain root module is no longer supported')
        old_verbose = langchain.verbose
    global _verbose
    return _verbose or old_verbose