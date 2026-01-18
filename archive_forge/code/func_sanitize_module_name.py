import os
import shutil
import string
from importlib import import_module
from pathlib import Path
from typing import Optional, cast
from urllib.parse import urlparse
import scrapy
from scrapy.commands import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.utils.template import render_templatefile, string_camelcase
def sanitize_module_name(module_name):
    """Sanitize the given module name, by replacing dashes and points
    with underscores and prefixing it with a letter if it doesn't start
    with one
    """
    module_name = module_name.replace('-', '_').replace('.', '_')
    if module_name[0] not in string.ascii_letters:
        module_name = 'a' + module_name
    return module_name