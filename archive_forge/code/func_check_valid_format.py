import numbers
import os
import sys
import warnings
from configparser import ConfigParser
from operator import itemgetter
from pathlib import Path
from typing import (
from scrapy.exceptions import ScrapyDeprecationWarning, UsageError
from scrapy.settings import BaseSettings
from scrapy.utils.deprecate import update_classpath
from scrapy.utils.python import without_none_values
def check_valid_format(output_format: str) -> None:
    if output_format not in valid_output_formats:
        raise UsageError(f"Unrecognized output format '{output_format}'. Set a supported one ({tuple(valid_output_formats)}) after a colon at the end of the output URI (i.e. -o/-O <URI>:<FORMAT>) or as a file extension.")