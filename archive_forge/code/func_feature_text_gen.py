from functools import partial
import json
import logging
import click
import cligj
from fiona.fio import helpers, options, with_context_env
from fiona.model import Geometry, ObjectEncoder
from fiona.transform import transform_geom
def feature_text_gen():
    yield first_line
    yield from stdin