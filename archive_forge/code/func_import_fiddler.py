import time
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import import_pandas
def import_fiddler() -> Any:
    """Import the fiddler python package and raise an error if it is not installed."""
    try:
        import fiddler
    except ImportError:
        raise ImportError('To use fiddler callback handler you need to have `fiddler-client`package installed. Please install it with `pip install fiddler-client`')
    return fiddler