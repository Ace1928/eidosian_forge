import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import srsly
from wasabi import MarkdownRenderer, Printer
from .. import about, util
from ..compat import importlib_metadata
from ._util import Arg, Opt, app, string_to_list
from .download import get_latest_version, get_model_filename
@app.command('info')
def info_cli(model: Optional[str]=Arg(None, help='Optional loadable spaCy pipeline'), markdown: bool=Opt(False, '--markdown', '-md', help='Generate Markdown for GitHub issues'), silent: bool=Opt(False, '--silent', '-s', '-S', help="Don't print anything (just return)"), exclude: str=Opt('labels', '--exclude', '-e', help='Comma-separated keys to exclude from the print-out'), url: bool=Opt(False, '--url', '-u', help='Print the URL to download the most recent compatible version of the pipeline')):
    """
    Print info about spaCy installation. If a pipeline is specified as an argument,
    print its meta information. Flag --markdown prints details in Markdown for easy
    copy-pasting to GitHub issues.

    Flag --url prints only the download URL of the most recent compatible
    version of the pipeline.

    DOCS: https://spacy.io/api/cli#info
    """
    exclude = string_to_list(exclude)
    info(model, markdown=markdown, silent=silent, exclude=exclude, url=url)