import logging
import os
import tarfile
import warnings
import zipfile
from . import _constants as C
from . import vocab
from ... import ndarray as nd
from ... import registry
from ... import base
from ...util import is_np_array
from ... import numpy as _mx_np
from ... import numpy_extension as _mx_npx
def get_pretrained_file_names(embedding_name=None):
    """Get valid token embedding names and their pre-trained file names.


    To load token embedding vectors from an externally hosted pre-trained token embedding file,
    such as those of GloVe and FastText, one should use
    `mxnet.contrib.text.embedding.create(embedding_name, pretrained_file_name)`.
    This method returns all the valid names of `pretrained_file_name` for the specified
    `embedding_name`. If `embedding_name` is set to None, this method returns all the valid
    names of `embedding_name` with their associated `pretrained_file_name`.


    Parameters
    ----------
    embedding_name : str or None, default None
        The pre-trained token embedding name.


    Returns
    -------
    dict or list:
        A list of all the valid pre-trained token embedding file names (`pretrained_file_name`)
        for the specified token embedding name (`embedding_name`). If the text embeding name is
        set to None, returns a dict mapping each valid token embedding name to a list of valid
        pre-trained files (`pretrained_file_name`). They can be plugged into
        `mxnet.contrib.text.embedding.create(embedding_name,
        pretrained_file_name)`.
    """
    text_embedding_reg = registry.get_registry(_TokenEmbedding)
    if embedding_name is not None:
        if embedding_name not in text_embedding_reg:
            raise KeyError('Cannot find `embedding_name` %s. Use `get_pretrained_file_names(embedding_name=None).keys()` to get all the valid embedding names.' % embedding_name)
        return list(text_embedding_reg[embedding_name].pretrained_file_name_sha1.keys())
    else:
        return {embedding_name: list(embedding_cls.pretrained_file_name_sha1.keys()) for embedding_name, embedding_cls in registry.get_registry(_TokenEmbedding).items()}