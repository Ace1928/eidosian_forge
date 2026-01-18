import re
import click
import json
from .instance import import_module
from ..interfaces.base import InputMultiPath, traits
from ..interfaces.base.support import get_trait_desc
def check_not_none(ctx, param, value):
    if value is None:
        raise click.BadParameter('got {}.'.format(value))
    return value