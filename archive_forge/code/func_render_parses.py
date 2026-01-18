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
def render_parses(docs: List[Doc], output_path: Path, model_name: str='', limit: int=250, deps: bool=True, ents: bool=True, spans: bool=True):
    docs[0].user_data['title'] = model_name
    if ents:
        html = displacy.render(docs[:limit], style='ent', page=True)
        with (output_path / 'entities.html').open('w', encoding='utf8') as file_:
            file_.write(html)
    if deps:
        html = displacy.render(docs[:limit], style='dep', page=True, options={'compact': True})
        with (output_path / 'parses.html').open('w', encoding='utf8') as file_:
            file_.write(html)
    if spans:
        html = displacy.render(docs[:limit], style='span', page=True)
        with (output_path / 'spans.html').open('w', encoding='utf8') as file_:
            file_.write(html)