from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
def bids_gen_info(bids_event_files, condition_column='', amplitude_column=None, time_repetition=False):
    """
    Generate a subject_info structure from a list of BIDS .tsv event files.

    Parameters
    ----------
    bids_event_files : list of str
        Filenames of BIDS .tsv event files containing columns including:
        'onset', 'duration', and 'trial_type' or the `condition_column` value.
    condition_column : str
        Column of files in `bids_event_files` based on the values of which
        events will be sorted into different regressors
    amplitude_column : str
        Column of files in `bids_event_files` based on the values of which
        to apply amplitudes to events. If unspecified, all events will be
        represented with an amplitude of 1.

    Returns
    -------
    subject_info: list of Bunch

    """
    info = []
    for bids_event_file in bids_event_files:
        with open(bids_event_file) as f:
            f_events = csv.DictReader(f, skipinitialspace=True, delimiter='\t')
            events = [{k: v for k, v in row.items()} for row in f_events]
        if not condition_column:
            condition_column = '_trial_type'
            for i in events:
                i.update({condition_column: 'ev0'})
        conditions = sorted(set([i[condition_column] for i in events]))
        runinfo = Bunch(conditions=[], onsets=[], durations=[], amplitudes=[])
        for condition in conditions:
            selected_events = [i for i in events if i[condition_column] == condition]
            onsets = [float(i['onset']) for i in selected_events]
            durations = [float(i['duration']) for i in selected_events]
            if time_repetition:
                decimals = math.ceil(-math.log10(time_repetition))
                onsets = [np.round(i, decimals) for i in onsets]
                durations = [np.round(i, decimals) for i in durations]
            runinfo.conditions.append(condition)
            runinfo.onsets.append(onsets)
            runinfo.durations.append(durations)
            try:
                amplitudes = [float(i[amplitude_column]) for i in selected_events]
                runinfo.amplitudes.append(amplitudes)
            except KeyError:
                runinfo.amplitudes.append([1] * len(onsets))
        info.append(runinfo)
    return info