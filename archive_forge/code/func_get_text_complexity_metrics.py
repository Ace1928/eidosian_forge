import logging
import os
import random
import string
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from langchain_core.utils import get_from_dict_or_env
from langchain_community.callbacks.utils import (
def get_text_complexity_metrics() -> List[str]:
    """Get the text complexity metrics from textstat."""
    return ['flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index', 'coleman_liau_index', 'automated_readability_index', 'dale_chall_readability_score', 'difficult_words', 'linsear_write_formula', 'gunning_fog', 'fernandez_huerta', 'szigriszt_pazos', 'gutierrez_polini', 'crawford', 'gulpease_index', 'osman']