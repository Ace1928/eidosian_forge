from collections import defaultdict
import glob
import json
import os
import warnings
from .metrics import Gauge
from .metrics_core import Metric
from .mmap_dict import MmapedDict
from .samples import Sample
from .utils import floatToGoString
def mark_process_dead(pid, path=None):
    """Do bookkeeping for when one process dies in a multi-process setup."""
    if path is None:
        path = os.environ.get('PROMETHEUS_MULTIPROC_DIR', os.environ.get('prometheus_multiproc_dir'))
    for mode in _LIVE_GAUGE_MULTIPROCESS_MODES:
        for f in glob.glob(os.path.join(path, f'gauge_{mode}_{pid}.db')):
            os.remove(f)