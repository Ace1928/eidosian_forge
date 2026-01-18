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
def configure_follower(follower_ident):
    """Configure required state for a follower.

    This invokes in the parent process and typically includes
    database creation.

    """
    from sqlalchemy.testing import provision
    provision.FOLLOWER_IDENT = follower_ident