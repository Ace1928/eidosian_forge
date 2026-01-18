from itertools import chain
from pathlib import Path
from typing import Iterable, List, Optional, Union, cast
import srsly
import tqdm
from wasabi import msg
from ..tokens import Doc, DocBin
from ..util import ensure_path, load_model
from ..vocab import Vocab
from ._util import Arg, Opt, app, import_code, setup_gpu, walk_directory
@app.command('apply')
def apply_cli(model: str=Arg(..., help='Model name or path'), data_path: Path=Arg(..., help=path_help, exists=True), output_file: Path=Arg(..., help=out_help, dir_okay=False), code_path: Optional[Path]=Opt(None, '--code', '-c', help=code_help), text_key: str=Opt('text', '--text-key', '-tk', help='Key containing text string for JSONL'), force_overwrite: bool=Opt(False, '--force', '-F', help='Force overwriting the output file'), use_gpu: int=Opt(-1, '--gpu-id', '-g', help='GPU ID or -1 for CPU.'), batch_size: int=Opt(1, '--batch-size', '-b', help='Batch size.'), n_process: int=Opt(1, '--n-process', '-n', help='number of processors to use.')):
    """
    Apply a trained pipeline to documents to get predictions.
    Expects a loadable spaCy pipeline and path to the data, which
    can be a directory or a file.
    The data files can be provided in multiple formats:
        1. .spacy files
        2. .jsonl files with a specified "field" to read the text from.
        3. Files with any other extension are assumed to be containing
           a single document.
    DOCS: https://spacy.io/api/cli#apply
    """
    data_path = ensure_path(data_path)
    output_file = ensure_path(output_file)
    code_path = ensure_path(code_path)
    if output_file.exists() and (not force_overwrite):
        msg.fail(force_msg, exits=1)
    if not data_path.exists():
        msg.fail(f"Couldn't find data path: {data_path}", exits=1)
    import_code(code_path)
    setup_gpu(use_gpu)
    apply(data_path, output_file, model, text_key, batch_size, n_process)