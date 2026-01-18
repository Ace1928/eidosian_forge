from __future__ import annotations
import contextlib
import copy
import math
import re
import types
from enum import Enum, EnumMeta, auto
from typing import (
from typing_extensions import TypeAlias, TypeGuard
import streamlit as st
from streamlit import config, errors
from streamlit import logger as _logger
from streamlit import string_util
from streamlit.errors import StreamlitAPIException
def is_keras_model(obj: object) -> bool:
    """True if input looks like a Keras model."""
    return is_type(obj, 'keras.engine.sequential.Sequential') or is_type(obj, 'keras.engine.training.Model') or is_type(obj, 'tensorflow.python.keras.engine.sequential.Sequential') or is_type(obj, 'tensorflow.python.keras.engine.training.Model')