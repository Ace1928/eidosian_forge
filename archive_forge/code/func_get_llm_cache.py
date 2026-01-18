import warnings
from typing import TYPE_CHECKING, Optional
def get_llm_cache() -> 'BaseCache':
    """Get the value of the `llm_cache` global setting."""
    import langchain
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Importing llm_cache from langchain root module is no longer supported')
        old_llm_cache = langchain.llm_cache
    global _llm_cache
    return _llm_cache or old_llm_cache