from typing import TYPE_CHECKING, Dict, Any, Tuple, Callable, List, Optional, IO, Union
import sys
from pathlib import Path
from spacy import util
from spacy.errors import Errors
from spacy.util import registry
from wasabi import Printer
import tqdm
def console_logger_v2(progress_bar: bool=False, console_output: bool=True, output_file: Optional[Union[str, Path]]=None):
    """The ConsoleLogger.v2 prints out training logs in the console and/or saves them to a jsonl file.
    progress_bar (bool): Whether the logger should print a progress bar tracking the steps till the next evaluation pass.
    console_output (bool): Whether the logger should print the logs on the console.
    output_file (Optional[Union[str, Path]]): The file to save the training logs to.
    """
    console_logger = registry.get('loggers', 'spacy.ConsoleLogger.v3')
    return console_logger(progress_bar=None if progress_bar is False else 'eval', console_output=console_output, output_file=output_file)