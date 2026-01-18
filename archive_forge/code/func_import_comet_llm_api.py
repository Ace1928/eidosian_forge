from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict
from langchain_core.tracers import BaseTracer
def import_comet_llm_api() -> SimpleNamespace:
    """Import comet_llm api and raise an error if it is not installed."""
    try:
        from comet_llm import experiment_info, flush
        from comet_llm.chains import api as chain_api
        from comet_llm.chains import chain, span
    except ImportError:
        raise ImportError('To use the CometTracer you need to have the `comet_llm>=2.0.0` python package installed. Please install it with `pip install -U comet_llm`')
    return SimpleNamespace(chain=chain, span=span, chain_api=chain_api, experiment_info=experiment_info, flush=flush)