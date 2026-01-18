from __future__ import print_function
from pandas import concat, read_csv
from argparse import ArgumentParser, FileType
from numpy import empty
def onsets_for(cond, run_df):
    """
    Inputs:
      * Condition Label to grab onsets, durations & amplitudes for.
      * Pandas Dataframe for current run containing onsets values as columns.

    Outputs:
      * Returns a dictionary of extracted values for onsets, durations, etc.
      * Returns None if there are no onsets.
    """
    condinfo = {}
    cond_df = run_df[run_df['condition'] == cond]
    if cond_df['onset'].notnull().any():
        if cond_df['duration'].notnull().any():
            durations = cond_df['duration'].tolist()
        else:
            durations = [0]
        condinfo = dict(name=cond, durations=durations, onsets=cond_df['onset'].tolist())
        if 'amplitude' in cond_df.columns and cond_df['amplitude'].notnull().any():
            pmods = [dict(name=args.pmod_name, poly=1, param=cond_df['amplitude'].tolist())]
            condinfo['pmod'] = pmods
    else:
        condinfo = None
    return condinfo