from pathlib import Path
from typing import Any, Callable, Dict, Set, Union
from xformers.utils import (
from .base import PositionEmbedding, PositionEmbeddingConfig  # noqa
from .rotary import RotaryEmbedding  # noqa
from .sine import SinePositionalEmbedding  # type: ignore  # noqa
from .vocab import VocabEmbedding  # noqa
import_all_modules(
def build_positional_embedding(config: Union[Dict[str, Any], PositionEmbeddingConfig]):
    """Builds a position encoding from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_position_encoding",
    "foo": "bar"}` will find a class that was registered as "my_position_encoding"
    (see :func:`register_positional_embedding`) and call .from_config on it."""
    if not isinstance(config, PositionEmbeddingConfig):
        config_instance = generate_matching_config(config, POSITION_EMBEDDING_REGISTRY[config['name']].config)
    else:
        config_instance = config
    return POSITION_EMBEDDING_REGISTRY[config_instance.name].constructor.from_config(config_instance)