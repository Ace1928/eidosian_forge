from __future__ import annotations
import sys
import openai
from .. import OpenAI, _load_client
from .._compat import model_json
from .._models import BaseModel
def print_model(model: BaseModel) -> None:
    sys.stdout.write(model_json(model, indent=2) + '\n')