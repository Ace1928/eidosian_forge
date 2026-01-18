import contextlib
import re
import tempfile
import numpy
import srsly
from thinc.api import get_current_ops
from spacy.tokens import Doc
from spacy.training import split_bilu_label
from spacy.util import make_tempdir  # noqa: F401
from spacy.vocab import Vocab
def get_cosine(vec1, vec2):
    """Get cosine for two given vectors"""
    OPS = get_current_ops()
    v1 = OPS.to_numpy(OPS.asarray(vec1))
    v2 = OPS.to_numpy(OPS.asarray(vec2))
    return numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))