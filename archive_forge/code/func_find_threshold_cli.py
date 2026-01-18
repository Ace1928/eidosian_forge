import functools
import logging
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy
import wasabi.tables
from .. import util
from ..errors import Errors
from ..pipeline import MultiLabel_TextCategorizer, TextCategorizer
from ..training import Corpus
from ._util import Arg, Opt, app, import_code, setup_gpu
@app.command('find-threshold', context_settings={'allow_extra_args': False, 'ignore_unknown_options': True})
def find_threshold_cli(model: str=Arg(..., help='Model name or path'), data_path: Path=Arg(..., help='Location of binary evaluation data in .spacy format', exists=True), pipe_name: str=Arg(..., help='Name of pipe to examine thresholds for'), threshold_key: str=Arg(..., help="Key of threshold attribute in component's configuration"), scores_key: str=Arg(..., help='Metric to optimize'), n_trials: int=Opt(_DEFAULTS['n_trials'], '--n_trials', '-n', help='Number of trials to determine optimal thresholds'), code_path: Optional[Path]=Opt(None, '--code', '-c', help='Path to Python file with additional code (registered functions) to be imported'), use_gpu: int=Opt(_DEFAULTS['use_gpu'], '--gpu-id', '-g', help='GPU ID or -1 for CPU'), gold_preproc: bool=Opt(_DEFAULTS['gold_preproc'], '--gold-preproc', '-G', help='Use gold preprocessing'), verbose: bool=Opt(False, '--verbose', '-V', '-VV', help='Display more information for debugging purposes')):
    """
    Runs prediction trials for a trained model with varying tresholds to maximize
    the specified metric. The search space for the threshold is traversed linearly
    from 0 to 1 in `n_trials` steps. Results are displayed in a table on `stdout`
    (the corresponding API call to `spacy.cli.find_threshold.find_threshold()`
    returns all results).

    This is applicable only for components whose predictions are influenced by
    thresholds - e.g. `textcat_multilabel` and `spancat`, but not `textcat`. Note
    that the full path to the corresponding threshold attribute in the config has to
    be provided.

    DOCS: https://spacy.io/api/cli#find-threshold
    """
    if verbose:
        util.logger.setLevel(logging.DEBUG)
    import_code(code_path)
    find_threshold(model=model, data_path=data_path, pipe_name=pipe_name, threshold_key=threshold_key, scores_key=scores_key, n_trials=n_trials, use_gpu=use_gpu, gold_preproc=gold_preproc, silent=False)