import sys
from typing import Dict, Any, Tuple, Callable, Iterator, List, Optional, IO
import re
from spacy import Language
from spacy.util import registry
def setup_default_console_logger(nlp: 'Language', stdout: IO=sys.stdout, stderr: IO=sys.stderr) -> Tuple[Callable, Callable]:
    console_logger = registry.get('loggers', 'spacy.ConsoleLogger.v1')
    console = console_logger(progress_bar=False)
    console_log_step, console_finalize = console(nlp, stdout, stderr)
    return (console_log_step, console_finalize)