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
@property
def templates_dir(self) -> str:
    return str(Path(self.settings['TEMPLATES_DIR'] or Path(scrapy.__path__[0], 'templates'), 'spiders'))