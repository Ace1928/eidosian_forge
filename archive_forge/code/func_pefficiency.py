from __future__ import annotations
import collections
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
def pefficiency(self):
    """
        Analyze the parallel efficiency.

        Returns:
            ParallelEfficiency object.
        """
    timers = self.timers()
    ncpus = [timer.ncpus for timer in timers]
    min_idx = np.argmin(ncpus)
    min_ncpus = ncpus[min_idx]
    ref_t = timers[min_idx]
    peff = {}
    ctime_peff = [min_ncpus * ref_t.wall_time / (t.wall_time * ncp) for t, ncp in zip(timers, ncpus)]
    wtime_peff = [min_ncpus * ref_t.cpu_time / (t.cpu_time * ncp) for t, ncp in zip(timers, ncpus)]
    n = len(timers)
    peff['total'] = {}
    peff['total']['cpu_time'] = ctime_peff
    peff['total']['wall_time'] = wtime_peff
    peff['total']['cpu_fract'] = n * [100]
    peff['total']['wall_fract'] = n * [100]
    for sect_name in self.section_names():
        ref_sect = ref_t.get_section(sect_name)
        sects = [timer.get_section(sect_name) for timer in timers]
        try:
            ctime_peff = [min_ncpus * ref_sect.cpu_time / (s.cpu_time * ncp) for s, ncp in zip(sects, ncpus)]
            wtime_peff = [min_ncpus * ref_sect.wall_time / (s.wall_time * ncp) for s, ncp in zip(sects, ncpus)]
        except ZeroDivisionError:
            ctime_peff = n * [-1]
            wtime_peff = n * [-1]
        assert sect_name not in peff
        peff[sect_name] = {}
        peff[sect_name]['cpu_time'] = ctime_peff
        peff[sect_name]['wall_time'] = wtime_peff
        peff[sect_name]['cpu_fract'] = [s.cpu_fract for s in sects]
        peff[sect_name]['wall_fract'] = [s.wall_fract for s in sects]
    return ParallelEfficiency(self._filenames, min_idx, peff)