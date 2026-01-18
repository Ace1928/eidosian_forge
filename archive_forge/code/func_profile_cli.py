import cProfile
import itertools
import pstats
import sys
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union
import srsly
import tqdm
import typer
from wasabi import Printer, msg
from ..language import Language
from ..util import load_model
from ._util import NAME, Arg, Opt, app, debug_cli
@debug_cli.command('profile')
@app.command('profile', hidden=True)
def profile_cli(ctx: typer.Context, model: str=Arg(..., help='Trained pipeline to load'), inputs: Optional[Path]=Arg(None, help="Location of input file. '-' for stdin.", exists=True, allow_dash=True), n_texts: int=Opt(10000, '--n-texts', '-n', help='Maximum number of texts to use if available')):
    """
    Profile which functions take the most time in a spaCy pipeline.
    Input should be formatted as one JSON object per line with a key "text".
    It can either be provided as a JSONL file, or be read from sys.sytdin.
    If no input file is specified, the IMDB dataset is loaded via Thinc.

    DOCS: https://spacy.io/api/cli#debug-profile
    """
    if ctx.parent.command.name == NAME:
        msg.warn("The profile command is now available via the 'debug profile' subcommand. You can run python -m spacy debug --help for an overview of the other available debugging commands.")
    profile(model, inputs=inputs, n_texts=n_texts)