import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import srsly
from thinc.api import fix_random_seed
from wasabi import Printer
from .. import displacy, util
from ..scorer import Scorer
from ..tokens import Doc
from ..training import Corpus
from ._util import Arg, Opt, app, benchmark_cli, import_code, setup_gpu
def print_textcats_auc_per_cat(msg: Printer, scores: Dict[str, Dict[str, float]]) -> None:
    msg.table([(k, f'{v:.2f}' if isinstance(v, (float, int)) else v) for k, v in scores.items()], header=('', 'ROC AUC'), aligns=('l', 'r'), title='Textcat ROC AUC (per label)')