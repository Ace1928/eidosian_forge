from __future__ import annotations
import abc
from argparse import Namespace
import configparser
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any
from sqlalchemy.testing import asyncio
def pre_begin(opt):
    """things to set up early, before coverage might be setup."""
    global options
    options = opt
    for fn in pre_configure:
        fn(options, file_config)