import logging
from pathlib import Path
from typing import Optional
import srsly
import typer
from wasabi import msg
from .. import util
from ..language import Language
from ..training.initialize import convert_vectors, init_nlp
from ._util import (
@init_cli.command('labels', context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
def init_labels_cli(ctx: typer.Context, config_path: Path=Arg(..., help='Path to config file', exists=True, allow_dash=True), output_path: Path=Arg(..., help='Output directory for the labels'), code_path: Optional[Path]=Opt(None, '--code', '-c', help='Path to Python file with additional code (registered functions) to be imported'), verbose: bool=Opt(False, '--verbose', '-V', '-VV', help='Display more information for debugging purposes'), use_gpu: int=Opt(-1, '--gpu-id', '-g', help='GPU ID or -1 for CPU')):
    """Generate JSON files for the labels in the data. This helps speed up the
    training process, since spaCy won't have to preprocess the data to
    extract the labels."""
    if verbose:
        util.logger.setLevel(logging.DEBUG)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    overrides = parse_config_overrides(ctx.args)
    import_code(code_path)
    setup_gpu(use_gpu)
    with show_validation_error(config_path):
        config = util.load_config(config_path, overrides=overrides)
    with show_validation_error(hint_fill=False):
        nlp = init_nlp(config, use_gpu=use_gpu)
    _init_labels(nlp, output_path)