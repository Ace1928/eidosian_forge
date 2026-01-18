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
def get_num_hits_per_matchup(self):
    """
        Return the number of hits per matchup.
        """
    matchup_total_1_df = self.matchup_total_df.reset_index()
    matchup_total_2_df = matchup_total_1_df.rename(columns={'eval_choice_0': 'eval_choice_1', 'eval_choice_1': 'eval_choice_0'})
    self.num_hits_per_matchup_df = pd.concat([matchup_total_1_df, matchup_total_2_df], axis=0).pivot(index='eval_choice_0', columns='eval_choice_1', values='matchup_total').reindex(index=self.models_by_win_frac, columns=self.models_by_win_frac)
    return self.num_hits_per_matchup_df