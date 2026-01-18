from __future__ import annotations
import base64
import html
import logging
import os
import pathlib
import pickle
import random
import re
import string
from io import StringIO
from typing import Optional, Union
from packaging.version import Version
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
def add_pandas_profile(self, name: str, profile: str) -> CardTab:
    """
        Add a new tab representing the provided pandas profile to the card.

        Args:
            name: Name of the variable in the Jinja2 template.
            profile: HTML string to render profile in the step card.

        Returns:
            The updated card instance.
        """
    try:
        profile_iframe = "<iframe srcdoc='{src}' width='100%' height='500' frameborder='0'></iframe>".format(src=html.escape(profile))
    except Exception as e:
        profile_iframe = f'Unable to create data profile. Error found:\n{e}'
    self.add_html(name, profile_iframe)
    return self