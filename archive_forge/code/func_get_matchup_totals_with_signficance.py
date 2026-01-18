import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.stats import binom_test
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai import __file__ as parlai_filepath
from parlai.core.params import ParlaiParser
import json
from IPython.core.display import HTML
def get_matchup_totals_with_signficance(self) -> pd.DataFrame:
    """
        Return dataframe with matchup win totals + significance.
        """

    def _signf_level(p):
        if p < 0.001:
            return ('***', 'p<.001')
        elif p < 0.01:
            return ('**', 'p<.01')
        elif p < 0.05:
            return ('*', 'p<.05')
        else:
            return ('', 'p>.05')
    output = []
    df_filtered = self.filter_by_dialogue_length()
    for _, run_annotations in df_filtered.groupby('run_id'):
        question = list(run_annotations.question)[0]
        for matchup, annotations in run_annotations.groupby('matchup'):
            model1, model2 = matchup.split('__vs__')
            wincount1 = np.sum(annotations['winner'] == model1)
            wincount2 = np.sum(annotations['winner'] == model2)
            winrate1 = np.mean(annotations['winner'] == model1)
            winrate2 = np.mean(annotations['winner'] == model2)
            p = binom_test([wincount1, wincount2])
            stars, plevel = _signf_level(p)
            agreements = []
            for _, pairing_annotations in annotations.groupby('pairing_id'):
                pair_wincount1 = np.sum(pairing_annotations['winner'] == model1)
                pair_wincount2 = np.sum(pairing_annotations['winner'] == model2)
                if pair_wincount1 < 2 and pair_wincount2 < 2:
                    if pair_wincount1 == 1 and pair_wincount2 == 1:
                        agreements.append(0)
                else:
                    majority_wincount = max(pair_wincount1, pair_wincount2)
                    num_pair_annotations = pair_wincount1 + pair_wincount2
                    pair_agreement = majority_wincount / num_pair_annotations
                    agreements.append(pair_agreement)
            total_agreement = np.mean(agreements)
            output.append({'question': question, 'matchup': matchup, 'model1': model1, 'model2': model2, 'numwins1': wincount1, 'numwins2': wincount2, 'winrate1': winrate1, 'winrate2': winrate2, 'p': p, 'stars': stars, 'sigf': plevel, 'agree': total_agreement})
    df_output = pd.DataFrame(output)
    self.signficance_df = df_output[['question', 'matchup', 'model1', 'numwins1', 'winrate1', 'model2', 'numwins2', 'winrate2', 'sigf', 'stars', 'p', 'agree']]
    return self.signficance_df