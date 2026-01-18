import random
import time
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Optional
import numpy
import typer
from tqdm import tqdm
from wasabi import msg
from .. import util
from ..language import Language
from ..tokens import Doc
from ..training import Corpus
from ._util import Arg, Opt, benchmark_cli, import_code, setup_gpu
@benchmark_cli.command('speed', context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
def benchmark_speed_cli(ctx: typer.Context, model: str=Arg(..., help='Model name or path'), data_path: Path=Arg(..., help='Location of binary evaluation data in .spacy format', exists=True), batch_size: Optional[int]=Opt(None, '--batch-size', '-b', min=1, help='Override the pipeline batch size'), no_shuffle: bool=Opt(False, '--no-shuffle', help='Do not shuffle benchmark data'), use_gpu: int=Opt(-1, '--gpu-id', '-g', help='GPU ID or -1 for CPU'), n_batches: int=Opt(50, '--batches', help='Minimum number of batches to benchmark', min=30), warmup_epochs: int=Opt(3, '--warmup', '-w', min=0, help='Number of iterations over the data for warmup'), code_path: Optional[Path]=Opt(None, '--code', '-c', help='Path to Python file with additional code (registered functions) to be imported')):
    """
    Benchmark a pipeline. Expects a loadable spaCy pipeline and benchmark
    data in the binary .spacy format.
    """
    import_code(code_path)
    setup_gpu(use_gpu=use_gpu, silent=False)
    nlp = util.load_model(model)
    batch_size = batch_size if batch_size is not None else nlp.batch_size
    corpus = Corpus(data_path)
    docs = [eg.predicted for eg in corpus(nlp)]
    if len(docs) == 0:
        msg.fail('Cannot benchmark speed using an empty corpus.', exits=1)
    print(f'Warming up for {warmup_epochs} epochs...')
    warmup(nlp, docs, warmup_epochs, batch_size)
    print()
    print(f'Benchmarking {n_batches} batches...')
    wps = benchmark(nlp, docs, n_batches, batch_size, not no_shuffle)
    print()
    print_outliers(wps)
    print_mean_with_ci(wps)