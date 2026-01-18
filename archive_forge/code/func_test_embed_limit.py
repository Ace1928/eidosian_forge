import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('method_name', [pytest.param('to_html5_video', marks=pytest.mark.skipif(not animation.writers.is_available(mpl.rcParams['animation.writer']), reason='animation writer not installed')), 'to_jshtml'])
@pytest.mark.parametrize('anim', [dict(frames=1)], indirect=['anim'])
def test_embed_limit(method_name, caplog, tmpdir, anim):
    caplog.set_level('WARNING')
    with tmpdir.as_cwd():
        with mpl.rc_context({'animation.embed_limit': 1e-06}):
            getattr(anim, method_name)()
    assert len(caplog.records) == 1
    record, = caplog.records
    assert record.name == 'matplotlib.animation' and record.levelname == 'WARNING'