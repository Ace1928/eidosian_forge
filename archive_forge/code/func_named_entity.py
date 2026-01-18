from the local database to W&B in Tables format.
import wandb
from wandb.integration.prodigy import upload_dataset
import base64
import collections.abc
import io
import urllib
from copy import deepcopy
import pandas as pd
from PIL import Image
import wandb
from wandb import util
from wandb.plots.utils import test_missing
from wandb.sdk.lib import telemetry as wb_telemetry
def named_entity(docs):
    """Create a named entity visualization.

    Taken from https://github.com/wandb/wandb/blob/main/wandb/plots/named_entity.py.
    """
    spacy = util.get_module('spacy', required='part_of_speech requires the spacy library, install with `pip install spacy`')
    util.get_module('en_core_web_md', required='part_of_speech requires `en_core_web_md` library, install with `python -m spacy download en_core_web_md`')
    if test_missing(docs=docs):
        html = spacy.displacy.render(docs, style='ent', page=True, minify=True, jupyter=False)
        wandb_html = wandb.Html(html)
        return wandb_html