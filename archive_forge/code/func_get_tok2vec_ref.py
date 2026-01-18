import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union
import srsly
from thinc.api import (
from thinc.config import ConfigValidationError
from wasabi import Printer
from ..errors import Errors
from ..schemas import ConfigSchemaPretrain
from ..tokens import Doc
from ..util import dot_to_object, load_model_from_config, registry
from .example import Example
def get_tok2vec_ref(nlp, pretrain_config):
    tok2vec_component = pretrain_config['component']
    if tok2vec_component is None:
        desc = f'To use pretrained tok2vec weights, [pretraining.component] needs to specify the component that should load them.'
        err = "component can't be null"
        errors = [{'loc': ['pretraining', 'component'], 'msg': err}]
        raise ConfigValidationError(config=nlp.config['pretraining'], errors=errors, desc=desc)
    layer = nlp.get_pipe(tok2vec_component).model
    if pretrain_config['layer']:
        layer = layer.get_ref(pretrain_config['layer'])
    return layer